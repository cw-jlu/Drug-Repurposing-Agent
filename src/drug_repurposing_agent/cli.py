"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import AgentInputs, RulePlanner, run_agent_task
from .deepseek import DeepSeekPlanner
from .workflow import Mode, run_expression_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditable expression-based drug repurposing")
    parser.add_argument("--question", help="Natural-language research request")
    parser.add_argument("--items", type=Path, help="Gene x drug CSV")
    parser.add_argument("--users", type=Path, help="Gene x disease CSV")
    parser.add_argument("--screen-dir", type=Path, help="Frozen LUAD screen directory")
    parser.add_argument("--disease-manifest", type=Path, help="GSE32863 disease manifest")
    parser.add_argument("--screen-manifest", type=Path, help="A549 screen manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=[x.value for x in Mode], default=Mode.BENCHMARK_STRICT.value)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--planner", choices=["rule", "deepseek"], default="rule")
    parser.add_argument("--deepseek-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    args = parser.parse_args()
    if args.question:
        try:
            planner = (RulePlanner(args.top_k) if args.planner == "rule"
                       else DeepSeekPlanner.from_env(args.deepseek_model))
        except ValueError as exc:
            parser.error(str(exc))
        result = run_agent_task(
            args.question,
            AgentInputs(args.items, args.users, args.screen_dir,
                        args.disease_manifest, args.screen_manifest),
            args.output,
            Mode(args.mode),
            planner,
        )
        print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                          "output": str(args.output), "plan": result.get("plan"),
                          "missing_inputs": result.get("missing_inputs")},
                         indent=2, ensure_ascii=False))
        return
    if args.items is None or args.users is None:
        parser.error("--items and --users are required unless --question is used")
    result = run_expression_workflow(args.items, args.users, args.output,
                                     Mode(args.mode), args.top_k)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "output": str(args.output), "qc": result["qc"]}, indent=2))
