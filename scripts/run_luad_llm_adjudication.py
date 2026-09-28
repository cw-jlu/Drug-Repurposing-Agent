"""Optional real DeepSeek review of the frozen LUAD Top-10 evidence matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.deepseek import DeepSeekConfig, DeepSeekPlanner, local_api_key
from drug_repurposing_agent.research_adjudication import adjudicate
from drug_repurposing_agent.trace import TraceRecorder


def local_key() -> str:
    return local_api_key()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path,
                        default=Path("configs/luad_top10_evidence_v1.json"))
    parser.add_argument("--matrix", type=Path,
                        default=Path("docs/luad_top10_evidence_matrix.md"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/reports/luad_llm_adjudication.json"))
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args()
    trace = TraceRecorder("luad_adjudication", args.output.parent / "traces")
    planner = None
    try:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        matrix = args.matrix.read_text(encoding="utf-8")
        hashes = {"evidence": sha256_file(args.evidence), "matrix": sha256_file(args.matrix)}
        trace.emit("input_loaded", input_paths={"evidence": str(args.evidence),
                    "matrix": str(args.matrix)}, input_sha256=hashes)
        key = local_key()
        trace.add_secret(key)
        planner = DeepSeekPlanner(key, DeepSeekConfig(
            model=args.model, base_url="https://api.deepseek.com/beta"))
        result = adjudicate(evidence, matrix, planner._post, model=args.model)
        result["evaluated_at"] = datetime.now(timezone.utc).isoformat()
        result["input_sha256"] = hashes
        result["trace_file"] = str(trace.path)
        result["provider_trace_file"] = planner.last_trace_path
        trace.emit("triage_validated", shortlist=result["shortlist"],
                   provider_trace=planner.last_trace_path, usage=result["provider_usage"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        trace.emit("output_saved", output=str(args.output))
    except Exception as exc:
        trace.emit("run_failed", error_type=type(exc).__name__, error=str(exc),
                   provider_trace=getattr(planner, "last_trace_path", None))
        raise
    print(f"Saved research-only adjudication to {args.output}")


if __name__ == "__main__":
    main()
