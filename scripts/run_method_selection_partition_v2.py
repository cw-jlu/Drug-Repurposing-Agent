"""Freeze repeated DeepSeek method choices before viewing partition outcomes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.deepseek import DeepSeekConfig, DeepSeekPlanner, local_api_key
from drug_repurposing_agent.model_selector import select_partition_method
from drug_repurposing_agent.trace import TraceRecorder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path,
                        default=Path("configs/method_selection_partition_v2_cases.json"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("configs/method_selection_partition_v2.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/reports/method_selection_partition_v2_choices.json"))
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args()
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
        rows = []
        for case in cases["cases"]:
            for repeat in range(1, protocol["model_repeats_per_partition"] + 1):
                response = select_partition_method(case["blind_input"], client._post, args.model)
                provider_trace = Path(client.last_trace_path or "")
                if not client.last_trace_path or not provider_trace.is_file():
                    raise ValueError("Provider-visible trace missing for a method choice")
                row = {"case_id": case["id"], "repeat": repeat, **response,
                       "provider_trace_file": client.last_trace_path,
                       "provider_trace_sha256": sha256_file(provider_trace)}
                rows.append(row)
                trace.emit("choice_validated", **row)
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
