"""Make one prospective DeepSeek choice before reading component results."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.deepseek import DeepSeekConfig, DeepSeekPlanner
from drug_repurposing_agent.model_selector import select_methods
from scripts.run_luad_llm_adjudication import local_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("configs/component_selector_v1.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/results/recess_component_llm_selector_v1.json"))
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args()
    config = json.loads(args.input.read_text(encoding="utf-8"))
    client = DeepSeekPlanner(local_key(), DeepSeekConfig(
        model=args.model, base_url="https://api.deepseek.com/beta"))
    result = select_methods(config, client._post, args.model)
    result["chosen_at"] = datetime.now(timezone.utc).isoformat()
    result["input_sha256"] = sha256_file(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved prescore method choices to {args.output}")


if __name__ == "__main__":
    main()
