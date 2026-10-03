"""Validate every registry disease end to end with the real tools (rule planner, no model calls).

For each registered disease the same request is run: fetch (if needed) -> cohort QC ->
signature from raw data -> Hallmark pathways -> ranking with the registered cell line ->
reference-drug audit -> report; literature review is excluded so no model is called
(it is covered by the LUAD frozen review and the live breast-cancer run). Reference-drug
lists were committed before any ranking for the three new diseases was computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from drug_repurposing_agent.agent_v2 import DeepSeekPlannerV2, RulePlannerV2, run_agent_v2
from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.geo_cohort import files_ready, load_registry
from drug_repurposing_agent.luad_tools_v2 import available_inputs, real_backend
from drug_repurposing_agent.trace import TraceRecorder, traced_run
from drug_repurposing_agent.workflow import Mode

OUTPUT = Path("benchmark/results/registry_validation_v1.json")
ROOT = Path("artifacts/agent_v2_runs/registry_validation")


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["rule", "deepseek"], default="rule")
    parser.add_argument("--full-request", action="store_true",
                        help="ask for the full evidence report (literature included) instead of excluding it")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = {}
    for disease_id, entry in load_registry().items():
        question = (f"请为{entry['aliases'][0]}筛选候选药物并给出证据报告" if args.full_request else
                    f"请为{entry['aliases'][0]}筛选候选药物并分析通路，不查文献，出报告")
        planner = DeepSeekPlannerV2.from_env() if args.planner == "deepseek" else RulePlannerV2()
        on_disk_before = files_ready(entry)
        started = perf_counter()
        out = ROOT / args.planner / disease_id
        report = run_agent_v2(question, Mode.RESEARCH_OPEN, available_inputs(question, live_review=False), planner,
                              real_backend(out, Mode.RESEARCH_OPEN, disease=entry), out)
        seconds = round(perf_counter() - started, 1)
        summaries = {e["tool"]: e.get("summary") for e in report["executed"] if e["status"] == "ok"}
        rows[disease_id] = {"label": entry["label"], "accession": entry["accession"], "design": entry.get("design", "paired"),
                            "cell_line": entry["drug_cell_line"], "question": question, "status": report["status"],
                            "files_on_disk_before_run": on_disk_before, "seconds": seconds,
                            "plan": [s["tool"] for r in report["rounds"] for s in (r.get("steps") or [])],
                            "plans_by_round": [[s["tool"] for s in (r.get("steps") or [])] for r in report["rounds"]],
                            "summaries": summaries,
                            "failures": [e for e in report["executed"] if e["status"] == "failed"]}
        trace.emit("disease_validated", disease=disease_id, status=report["status"], seconds=seconds)
        print(disease_id, report["status"], seconds, "s")
    result = {"eval_name": args.output.stem, "planner": args.planner, "full_request": args.full_request,
              "note": __doc__, "diseases": rows, "trace_file": str(trace.path)}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))


def main() -> None:
    traced_run("registry_validation_v1", _main)


if __name__ == "__main__":
    main()
