"""Run agent v2 end to end: natural-language request -> multi-step plan -> validated execution.

Example (repository root):
    python -m scripts.run_agent_v2 --question "请为肺腺癌筛选候选药物并给出证据报告" --planner deepseek
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from drug_repurposing_agent.agent_v2 import DeepSeekPlannerV2, RulePlannerV2, ToolError, run_agent_v2
from drug_repurposing_agent.luad_tools_v2 import available_inputs, real_backend
from drug_repurposing_agent.workflow import Mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.RESEARCH_OPEN.value)
    parser.add_argument("--planner", choices=["rule", "deepseek"], default="rule")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--live-review", action="store_true",
                        help="Run the multi-agent literature review live instead of reusing the frozen one")
    parser.add_argument("--inputs", default=None,
                        help="Comma-separated subset of the detected inputs (to test missing-input behaviour)")
    parser.add_argument("--inject-failure", default=None,
                        help="Demo only: TOOL:N makes the real TOOL fail its first N attempts")
    args = parser.parse_args()
    mode = Mode(args.mode)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("artifacts/agent_v2_runs") / f"{stamp}_{uuid4().hex[:8]}"
    planner = DeepSeekPlannerV2.from_env(args.model) if args.planner == "deepseek" else RulePlannerV2()
    backend = real_backend(output, mode, live_review=args.live_review)
    inputs = available_inputs()
    if args.inputs is not None:
        wanted = [x for x in args.inputs.split(",") if x]
        unknown = [x for x in wanted if x not in inputs]
        if unknown:
            parser.error(f"inputs not available on this machine: {unknown}")
        inputs = tuple(wanted)
    if args.inject_failure:
        tool, count = args.inject_failure.split(":")
        remaining = {tool: int(count)}
        real = backend

        def backend(step, artefacts):
            if remaining.get(step.tool, 0) > 0:
                remaining[step.tool] -= 1
                raise ToolError(f"injected demo failure of {step.tool}")
            return real(step, artefacts)
    report = run_agent_v2(args.question, mode, inputs, planner, backend, output)
    print(json.dumps({"status": report["status"], "rounds": len(report["rounds"]),
                      "executed": [(e["tool"], e["status"]) for e in report["executed"]],
                      "output": str(output)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
