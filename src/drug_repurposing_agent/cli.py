"""Command line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import Mode, run_expression_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditable expression-based drug repurposing")
    parser.add_argument("--items", type=Path, required=True, help="Gene x drug CSV")
    parser.add_argument("--users", type=Path, required=True, help="Gene x disease CSV")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=[x.value for x in Mode], default=Mode.BENCHMARK_STRICT.value)
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()
    result = run_expression_workflow(args.items, args.users, args.output,
                                     Mode(args.mode), args.top_k)
    print(json.dumps({"run_id": result["run_id"], "status": result["status"],
                      "output": str(args.output), "qc": result["qc"]}, indent=2))
