"""Make one prospective DeepSeek choice before reading component results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.deepseek import DeepSeekConfig, DeepSeekPlanner
from drug_repurposing_agent.model_selector import select_methods
from drug_repurposing_agent.trace import TraceRecorder
from scripts.run_luad_llm_adjudication import local_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("configs/component_selector_v1.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/reports/component_llm_selector.json"))
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args()
    trace = TraceRecorder("component_selector", args.output.parent / "traces")
    client = None
    try:
        config = json.loads(args.input.read_text(encoding="utf-8"))
        trace.emit("input_loaded", input_path=str(args.input), input_sha256=sha256_file(args.input),
                   config=config)
        key = local_key()
        trace.add_secret(key)
        client = DeepSeekPlanner(key, DeepSeekConfig(
            model=args.model, base_url="https://api.deepseek.com/beta"))
        result = select_methods(config, client._post, args.model)
        result["chosen_at"] = datetime.now(timezone.utc).isoformat()
        result["input_sha256"] = sha256_file(args.input)
        result["trace_file"] = str(trace.path)
        result["provider_trace_file"] = client.last_trace_path
        trace.emit("choice_validated", choices=result["choices"],
                   provider_trace=client.last_trace_path, usage=result["provider_usage"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        trace.emit("output_saved", output=str(args.output))
    except Exception as exc:
        trace.emit("run_failed", error_type=type(exc).__name__, error=str(exc),
                   provider_trace=getattr(client, "last_trace_path", None))
        raise
    print(f"Saved prescore method choices to {args.output}")


if __name__ == "__main__":
    main()
