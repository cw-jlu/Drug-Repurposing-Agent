"""Optional real DeepSeek review of the frozen LUAD Top-10 evidence matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.deepseek import DeepSeekConfig, DeepSeekPlanner
from drug_repurposing_agent.research_adjudication import adjudicate


def local_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if key:
        return key
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError("Set DEEPSEEK_API_KEY or create an ignored local .env file")


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
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    matrix = args.matrix.read_text(encoding="utf-8")
    planner = DeepSeekPlanner(local_key(), DeepSeekConfig(
        model=args.model, base_url="https://api.deepseek.com/beta"))
    result = adjudicate(evidence, matrix, planner._post, model=args.model)
    result["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    result["input_sha256"] = {"evidence": sha256_file(args.evidence),
                              "matrix": sha256_file(args.matrix)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved research-only adjudication to {args.output}")


if __name__ == "__main__":
    main()
