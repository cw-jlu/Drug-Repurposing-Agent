"""Grade planner evaluation traces without executing scientific tools or models."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from drug_repurposing_agent.agent import ToolCall, validate_tool_call
from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder
from drug_repurposing_agent.workflow import Mode


def _trace_path(value: str, report_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute() or path.is_file():
        return path
    return report_path.parent / path


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def grade_result(result: dict, cases: dict[str, dict], report_path: Path) -> dict:
    """Compare report claims with a complete local trace, case by case."""
    trace_value = result.get("trace_file")
    if not trace_value:
        raise ValueError("Evaluation has no trace_file; historical runs cannot be trace-graded")
    path = _trace_path(str(trace_value), report_path)
    events = _events(path)
    if not events or events[0].get("stage") != "trace_started" or events[-1].get("stage") != "evaluation_finished":
        raise ValueError("Evaluation trace is incomplete")
    run_ids = {event.get("run_id") for event in events}
    if len(run_ids) != 1 or None in run_ids or any(event.get("schema_version") != 1 for event in events):
        raise ValueError("Evaluation trace schema or run ID is inconsistent")
    starts = {event["case_id"]: event for event in events if event.get("stage") == "case_started"}
    finishes = {event["id"]: event for event in events if event.get("stage") == "case_finished"}
    proposed = {event["case_id"]: event for event in events if event.get("stage") == "case_plan_proposed"}
    records = result.get("records", [])
    expected_ids = [case["id"] for case in cases.values()]
    observed_ids = [record["id"] for record in records]
    if Counter(observed_ids) != Counter(expected_ids) or len(starts) != len(cases) or len(finishes) != len(cases):
        raise ValueError("Trace, report and case IDs do not match one-to-one")
    rows = []
    for record in records:
        case_id = record["id"]
        case = cases[case_id]
        start, finish = starts[case_id], finishes[case_id]
        calls = proposed.get(case_id, {}).get("calls", [])
        actual = record["actual_tool"]
        reported_match = (finish.get("actual_tool") == actual and
                          finish.get("expected_tool") == case["expected_tool"] and
                          finish.get("correct") == (actual == case["expected_tool"]))
        input_match = (start.get("question") == case["question"] and
                       start.get("mode") == case["mode"] and
                       start.get("available_inputs") == case["available_inputs"])
        argument_valid = False
        if len(calls) == 1 and calls[0].get("name") == actual and isinstance(calls[0].get("arguments"), dict):
            try:
                validate_tool_call(ToolCall(actual, calls[0]["arguments"]), Mode(case["mode"]))
                argument_valid = True
            except ValueError:
                pass
        tool_match = actual == case["expected_tool"]
        safe_boundary = (case["expected_tool"] != "manual_review" or actual == "manual_review")
        provider_trace = finish.get("provider_trace")
        provider_trace_exists = (provider_trace is None or
                                 _trace_path(str(provider_trace), report_path).is_file())
        passed = (reported_match and input_match and tool_match and argument_valid and
                  safe_boundary and provider_trace_exists and record.get("error_type") is None)
        rows.append({"id": case_id, "category": case.get("category"),
                     "expected_tool": case["expected_tool"], "actual_tool": actual,
                     "tool_match": tool_match, "argument_valid": argument_valid,
                     "safe_boundary": safe_boundary, "input_match": input_match,
                     "report_trace_match": reported_match,
                     "provider_trace_exists": provider_trace_exists,
                     "error_type": record.get("error_type"), "passed": passed})
    return {"planner": result["planner"], "trace_file": str(path),
            "trace_sha256": sha256_file(path), "cases": len(rows),
            "passed": sum(row["passed"] for row in rows),
            "tool_correct": sum(row["tool_match"] for row in rows),
            "arguments_valid": sum(row["argument_valid"] for row in rows),
            "safe_boundaries": sum(row["safe_boundary"] for row in rows),
            "rows": rows}


def grade_report(report_path: Path, cases_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    definition = json.loads(cases_path.read_text(encoding="utf-8"))
    if report.get("case_file_sha256") != sha256_file(cases_path):
        raise ValueError("Case file differs from evaluated input")
    cases = {case["id"]: case for case in definition["cases"]}
    if len(cases) != len(definition["cases"]):
        raise ValueError("Duplicate case ID")
    return {"graded_at": datetime.now(timezone.utc).isoformat(),
            "source_report": str(report_path), "source_report_sha256": sha256_file(report_path),
            "case_file": str(cases_path), "case_file_sha256": sha256_file(cases_path),
            "results": [grade_result(result, cases, report_path)
                        for result in report["results"]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/reports/planner_trace_grade.json"))
    args = parser.parse_args()
    trace = TraceRecorder("planner_trace_grading", args.output.parent / "traces")
    trace.emit("grading_started", source_report=str(args.report), cases=str(args.cases))
    try:
        result = grade_report(args.report, args.cases)
        result["trace_file"] = str(trace.path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        trace.emit("grading_completed", output=str(args.output),
                   summaries=[{"planner": row["planner"], "passed": row["passed"],
                               "cases": row["cases"]} for row in result["results"]])
        for row in result["results"]:
            print(f"{row['planner']}: trace-grade {row['passed']}/{row['cases']}")
    except Exception as exc:
        trace.emit("grading_failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
