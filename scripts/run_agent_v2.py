"""Run agent v2 end to end: natural-language request -> multi-step plan -> validated execution.

Example (repository root):
    python -m scripts.run_agent_v2 --question "请为肺腺癌筛选候选药物并给出证据报告"

The DeepSeek planner is the default; without a DeepSeek key (environment or ignored
.env) the run falls back to the rule planner and says so. ``--planner rule`` forces it.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from drug_repurposing_agent.agent_v2 import DeepSeekPlannerV2, RulePlannerV2, ToolError, run_agent_v2
from drug_repurposing_agent.geo_cohort import load_registry, resolve_disease
from drug_repurposing_agent.luad_tools_v2 import available_inputs, real_backend
from drug_repurposing_agent.workflow import Mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--mode", choices=[m.value for m in Mode], default=Mode.RESEARCH_OPEN.value)
    parser.add_argument("--planner", choices=["rule", "deepseek"], default="deepseek")
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
    if args.planner == "deepseek":
        try:
            planner = DeepSeekPlannerV2.from_env(args.model)
        except ValueError:
            print("注意：没有找到 DeepSeek 密钥，改用规则规划器（--planner rule）。")
            planner = RulePlannerV2()
    else:
        planner = RulePlannerV2()
    registry = load_registry()
    disease = resolve_disease(args.question, registry)
    if disease is None:
        names = "、".join(e["label"] for e in registry.values())
        print(f"注意：请求中没有登记过的疾病（已登记：{names}），疾病分析工具不可用。")
    backend = real_backend(output, mode, disease=disease, live_review=args.live_review)
    inputs = available_inputs(args.question, live_review=args.live_review)
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
    print(json.dumps({"disease": disease["id"] if disease else None, "status": report["status"],
                      "rounds": len(report["rounds"]),
                      "executed": [(e["tool"], e["status"]) for e in report["executed"]],
                      "output": str(output)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
