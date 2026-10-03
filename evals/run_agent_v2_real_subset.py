"""Re-run representative frozen planning cases with the REAL tool backend.

Cases come unchanged from the frozen configs/planner_eval_multistep_v1.json (hash
checked) and are graded with the same grade() function. Failure-injection cases
are excluded because real tools cannot inject case-specific faults (those are
covered by the simulated eval and the archived observe-replan demo). The literature
tool reuses the frozen review (no live review), so no extra review calls occur.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from drug_repurposing_agent.agent_v2 import DeepSeekPlannerV2, RulePlannerV2, run_agent_v2
from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.luad_tools_v2 import available_inputs, default_luad, real_backend
from drug_repurposing_agent.trace import TraceRecorder, traced_run
from drug_repurposing_agent.workflow import Mode
from evals.run_planner_eval_multistep import FROZEN_SHA256, grade, load_cases

SUBSET = ("P01", "P02", "M01", "M05", "U01", "U06", "B01")


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["deepseek", "rule"], default="deepseek")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/agent_v2_real_subset.json"))
    args = parser.parse_args()
    cases = {c["id"]: c for c in load_cases()}
    machine = set(available_inputs())
    rows = []
    for cid in SUBSET:
        case = cases[cid]
        if not set(case["available_inputs"]) <= machine:
            raise RuntimeError(f"{cid} needs inputs not present on this machine")
        mode = Mode(case["mode"])
        out = Path("artifacts/agent_v2_runs/real_subset") / args.planner / cid
        planner = DeepSeekPlannerV2.from_env() if args.planner == "deepseek" else RulePlannerV2()
        report = run_agent_v2(case["question"], mode, tuple(case["available_inputs"]), planner,
                              real_backend(out, mode, disease=default_luad()), out)
        g = grade(case, report)
        rows.append({"id": cid, "category": case["category"], "question": case["question"], **g,
                     "plans": [[s["tool"] for s in (r.get("steps") or [])] for r in report["rounds"]],
                     "artefact_summaries": report.get("artefact_summaries", {})})
        trace.emit("case_done", id=cid, passed=g["passed"], status=g["status"])
        print(cid, "PASS" if g["passed"] else "FAIL", g["status"], g["tools_ok"], g["problems"])
    result = {"eval_name": "agent_v2_real_subset", "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "case_file_sha256_lf": FROZEN_SHA256, "backend": "real", "planner": args.planner,
              "subset": list(SUBSET), "passed": sum(r["passed"] for r in rows), "cases": rows,
              "note": __doc__, "trace_file": str(trace.path)}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))
    print("passed", result["passed"], "/", len(rows))


def main() -> None:
    traced_run("agent_v2_real_subset", _main)


if __name__ == "__main__":
    main()
