"""Freeze repeated DeepSeek method choices before viewing partition outcomes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.deepseek import DeepSeekConfig, DeepSeekPlanner, local_api_key
from drug_repurposing_agent.model_selector import select_partition_method
from drug_repurposing_agent.trace import TraceRecorder, verify_trace_chain


def preserved_trace(value: str) -> Path:
    supplied = Path(value)
    for candidate in (supplied, Path("benchmark/results/traces") / supplied.name,
                      Path("artifacts/traces") / supplied.name):
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise FileNotFoundError(f"Preserved provider trace missing: {supplied.name}")


def portable_trace_path(path: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        return str(resolved.relative_to(Path.cwd().resolve(strict=True)))
    except ValueError:
        return str(resolved)


def recover_first_choice(path: Path, blind_input: dict, model: str) -> dict:
    """Revalidate the first pre-outcome provider response without resampling it."""
    events, _ = verify_trace_chain(path, require_chain=True)
    if [event.get("stage") for event in events] != [
            "trace_started", "model_request", "model_response"]:
        raise ValueError("Recovery requires one complete, unused provider response")
    request = events[1]["payload"]
    response = events[2]["response"]

    def replay(payload: dict) -> dict:
        if payload != request:
            raise ValueError("Recovered request differs from the current frozen blind input")
        return response

    return select_partition_method(blind_input, replay, model)


def recover_attempt_rows(path: Path, cases: list[dict], repeats: int,
                         model: str) -> list[dict]:
    """Replay every provider response in a failed pre-outcome attempt, in order."""
    events, _ = verify_trace_chain(path, require_chain=True)
    stages = [event.get("stage") for event in events]
    if (stages[:2] != ["trace_started", "selection_started"] or
            stages[-1] != "selection_failed" or
            events[-1].get("error") != "Invalid partition choice or reason" or
            any(stage not in {"trace_started", "selection_started", "choice_validated",
                              "first_choice_recovered_after_reason_cap_amendment",
                              "selection_failed"} for stage in stages) or
            "selection_saved" in stages):
        raise ValueError("Recovery requires the complete failed prescore attempt")
    ordered = [(case, repeat) for case in cases for repeat in range(1, repeats + 1)]
    validated = [event for event in events if event.get("stage") == "choice_validated"]
    if not validated or len(validated) >= len(ordered):
        raise ValueError("Recovery requires a nonempty incomplete choice prefix")
    attempt_hash = sha256_file(path)
    rows = []
    seen = set()
    for index, (case, repeat) in enumerate(ordered[:len(validated) + 1]):
        prior = validated[index] if index < len(validated) else events[-1]
        if index < len(validated) and (prior.get("case_id"), prior.get("repeat")) != (case["id"], repeat):
            raise ValueError("Attempt choice order differs from frozen cases")
        provider_trace = preserved_trace(prior["provider_trace_file"] if index < len(validated)
                                         else prior["provider_trace"])
        if provider_trace in seen:
            raise ValueError("Attempt reused a provider trace")
        seen.add(provider_trace)
        if index < len(validated) and prior.get("provider_trace_sha256") != sha256_file(provider_trace):
            raise ValueError("Attempt provider trace hash differs")
        response = recover_first_choice(provider_trace, case["blind_input"], model)
        if index < len(validated) and any(prior.get(key) != response[key]
                                         for key in ("choice", "model", "usage")):
            raise ValueError("Attempt validated choice differs from provider response")
        rows.append({"case_id": case["id"], "repeat": repeat, **response,
                     "selection_source": "recovered_previous_attempt",
                     "recovery_attempt_trace_sha256": attempt_hash,
                     "provider_trace_file": portable_trace_path(provider_trace),
                     "provider_trace_sha256": sha256_file(provider_trace)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path,
                        default=Path("configs/method_selection_partition_v2_cases.json"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("configs/method_selection_partition_v2.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/reports/method_selection_partition_v2_choices.json"))
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--recover-first-provider-trace", type=Path,
                        help="Reuse one complete pre-outcome first response after format-only validation amendment")
    parser.add_argument("--recover-attempt-trace", type=Path,
                        help="Replay the complete failed pre-outcome attempt without resampling its responses")
    args = parser.parse_args()
    if args.recover_first_provider_trace and args.recover_attempt_trace:
        parser.error("Choose only one recovery source")
    trace = TraceRecorder("method_selection_v2", args.output.parent / "traces")
    client = None
    trace.emit("selection_started", cases=str(args.cases), protocol=str(args.protocol))
    try:
        if args.output.exists():
            raise ValueError("Choice output already exists; refusing to overwrite a frozen selection")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        if (cases.get("protocol_sha256") != sha256_file(args.protocol) or
                len(cases.get("cases", [])) != protocol["partition_count"]):
            raise ValueError("Cases do not match the frozen protocol")
        for case in cases["cases"]:
            case_root = Path(case["dataset_dir"]).parents[1]
            if (case_root / "b2").exists() or (case_root / "components").exists():
                raise ValueError("Partition outcomes may already exist; prescore selection is closed")
        key = local_api_key()
        trace.add_secret(key)
        client = DeepSeekPlanner(key, DeepSeekConfig(
            model=args.model, base_url="https://api.deepseek.com/beta"))
        rows = (recover_attempt_rows(args.recover_attempt_trace.resolve(strict=True),
                                     cases["cases"], protocol["model_repeats_per_partition"],
                                     args.model) if args.recover_attempt_trace else [])
        for row in rows:
            trace.emit("choice_recovered_from_failed_attempt", case_id=row["case_id"],
                       repeat=row["repeat"], provider_trace_sha256=row["provider_trace_sha256"],
                       attempt_trace_sha256=row["recovery_attempt_trace_sha256"])
            trace.emit("choice_validated", **row)
        recovered_count = len(rows)
        choice_index = 0
        for case_index, case in enumerate(cases["cases"]):
            for repeat in range(1, protocol["model_repeats_per_partition"] + 1):
                if choice_index < recovered_count:
                    choice_index += 1
                    continue
                recovered = (case_index == 0 and repeat == 1 and
                             args.recover_first_provider_trace is not None)
                if recovered:
                    provider_trace = preserved_trace(str(args.recover_first_provider_trace))
                    response = recover_first_choice(provider_trace, case["blind_input"], args.model)
                    trace.emit("first_choice_recovered_after_reason_cap_amendment",
                               case_id=case["id"], repeat=repeat,
                               provider_trace=str(provider_trace),
                               provider_trace_sha256=sha256_file(provider_trace),
                               local_reason_cap_before=200, local_reason_cap_after=1000)
                else:
                    response = select_partition_method(case["blind_input"], client._post,
                                                       args.model)
                    provider_trace = Path(client.last_trace_path or "")
                if not provider_trace.is_file():
                    raise ValueError("Provider-visible trace missing for a method choice")
                row = {"case_id": case["id"], "repeat": repeat, **response,
                       "selection_source": ("recovered_first_provider_response" if recovered
                                            else "live_provider_call"),
                       "provider_trace_file": portable_trace_path(provider_trace),
                       "provider_trace_sha256": sha256_file(provider_trace)}
                rows.append(row)
                trace.emit("choice_validated", **row)
                choice_index += 1
        report = {"status": "prescore_choices_frozen_outcomes_unseen",
                  "chosen_at": datetime.now(timezone.utc).isoformat(),
                  "protocol_sha256": sha256_file(args.protocol),
                  "cases_sha256": sha256_file(args.cases),
                  "model": args.model, "choices": rows,
                  "trace_file": str(trace.path)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"))
        trace.emit("selection_saved", output=str(args.output),
                   output_sha256=sha256_file(args.output), choices=len(rows))
        print(f"Saved {len(rows)} prescore choices to {args.output}")
    except Exception as exc:
        trace.emit("selection_failed", error_type=type(exc).__name__, error=str(exc),
                   provider_trace=getattr(client, "last_trace_path", None))
        raise


if __name__ == "__main__":
    main()
