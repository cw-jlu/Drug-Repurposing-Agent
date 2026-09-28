"""Audit every prospective selector decision against its provider-visible trace."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.model_selector import METHODS
from drug_repurposing_agent.trace import TraceRecorder, verify_trace_chain


def resolve_trace(value: str, report_path: Path) -> Path:
    supplied = Path(value)
    if supplied.is_file():
        return supplied.resolve(strict=True)
    candidate = report_path.parent / supplied
    if candidate.is_file():
        return candidate.resolve(strict=True)
    raise FileNotFoundError(f"Trace does not exist: {value}")


def read_trace(path: Path, *, first: str, last: str) -> list[dict]:
    events, _ = verify_trace_chain(path, require_chain=True)
    if (not events or events[0].get("stage") != first or events[-1].get("stage") != last or
            any(event.get("schema_version") != 1 for event in events) or
            len({event.get("run_id") for event in events}) != 1 or
            events[0].get("run_id") is None):
        raise ValueError(f"Incomplete or inconsistent trace: {path}")
    return events


def grade_choices(report_path: Path, cases_path: Path, protocol_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (report.get("status") != "prescore_choices_frozen_outcomes_unseen" or
            report.get("protocol_sha256") != sha256_file(protocol_path) or
            report.get("cases_sha256") != sha256_file(cases_path) or
            cases.get("protocol_sha256") != sha256_file(protocol_path)):
        raise ValueError("Report or cases do not match the frozen protocol")
    case_map = {case["id"]: case for case in cases["cases"]}
    if len(case_map) != protocol["partition_count"]:
        raise ValueError("Partition inventory differs from protocol")
    expected = Counter((case_id, repeat) for case_id in case_map
                       for repeat in range(1, protocol["model_repeats_per_partition"] + 1))
    rows = report.get("choices", [])
    observed = Counter((row.get("case_id"), row.get("repeat")) for row in rows)
    if observed != expected:
        raise ValueError("Each partition/repeat must have exactly one frozen choice")

    main_path = resolve_trace(report["trace_file"], report_path)
    main_events = read_trace(main_path, first="trace_started", last="selection_saved")
    validation_events = [event for event in main_events if event.get("stage") == "choice_validated"]
    if len(validation_events) != len(rows):
        raise ValueError("Main trace does not contain every selection")
    saved = main_events[-1]
    if saved.get("output_sha256") != sha256_file(report_path) or saved.get("choices") != len(rows):
        raise ValueError("Selection save receipt differs from frozen report")
    seen_provider_paths = set()
    grade_rows = []
    for row, event in zip(rows, validation_events, strict=True):
        expected_event = {key: value for key, value in row.items()}
        trace_match = all(event.get(key) == value for key, value in expected_event.items())
        provider_path = resolve_trace(row["provider_trace_file"], report_path)
        if provider_path in seen_provider_paths:
            raise ValueError("Provider trace reused for more than one choice")
        seen_provider_paths.add(provider_path)
        provider_events = read_trace(provider_path, first="trace_started", last="model_response")
        provider_hash_match = row.get("provider_trace_sha256") == sha256_file(provider_path)
        request_events = [item for item in provider_events if item.get("stage") == "model_request"]
        response_events = [item for item in provider_events if item.get("stage") == "model_response"]
        if len(request_events) != 1 or len(response_events) != 1:
            raise ValueError("Provider trace must contain exactly one request and response")
        payload = request_events[0].get("payload", {})
        response = response_events[0].get("response", {})
        request_match = False
        choice_match = False
        try:
            messages = payload["messages"]
            request_match = (payload["model"] == report["model"] and
                             json.loads(messages[-1]["content"]) ==
                             case_map[row["case_id"]]["blind_input"] and
                             payload["tools"][0]["function"]["name"] ==
                             "submit_partition_choice" and
                             payload["tool_choice"] == "required")
            call = response["choices"][0]["message"]["tool_calls"]
            choice_match = (len(call) == 1 and
                            call[0]["function"]["name"] == "submit_partition_choice" and
                            json.loads(call[0]["function"]["arguments"]) == row["choice"] and
                            row["choice"]["method"] in METHODS)
        except (KeyError, IndexError, TypeError, ValueError):
            pass
        passed = trace_match and provider_hash_match and request_match and choice_match
        grade_rows.append({"case_id": row["case_id"], "repeat": row["repeat"],
                           "method": row["choice"]["method"],
                           "main_trace_match": trace_match,
                           "provider_hash_match": provider_hash_match,
                           "blind_request_match": request_match,
                           "provider_choice_match": choice_match,
                           "provider_trace_sha256": sha256_file(provider_path),
                           "passed": passed})
    return {"status": "provider_visible_choice_trace_audit",
            "graded_at": datetime.now(timezone.utc).isoformat(),
            "choice_report_sha256": sha256_file(report_path),
            "cases_sha256": sha256_file(cases_path),
            "protocol_sha256": sha256_file(protocol_path),
            "main_trace_sha256": sha256_file(main_path),
            "choices": len(grade_rows),
            "passed": sum(row["passed"] for row in grade_rows),
            "rows": grade_rows,
            "limitation": ("Checks provider-visible request/response and saved choice, "
                           "not private reasoning or predictive validity.")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--choices", type=Path,
                        default=Path("artifacts/reports/method_selection_partition_v2_choices.json"))
    parser.add_argument("--cases", type=Path,
                        default=Path("configs/method_selection_partition_v2_cases.json"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("configs/method_selection_partition_v2.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/reports/method_selection_partition_v2_trace_grade.json"))
    args = parser.parse_args()
    trace = TraceRecorder("method_selection_trace_grading", args.output.parent / "traces")
    trace.emit("grading_started", choice_report=str(args.choices), cases=str(args.cases))
    try:
        result = grade_choices(args.choices, args.cases, args.protocol)
        result["trace_file"] = str(trace.path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8"))
        trace.emit("grading_completed", output=str(args.output),
                   output_sha256=sha256_file(args.output),
                   passed=result["passed"], choices=result["choices"])
        print(f"Provider-visible choice trace: {result['passed']}/{result['choices']}")
    except Exception as exc:
        trace.emit("grading_failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
