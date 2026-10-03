"""Validate every registry disease end to end with the real tools (rule planner, no model calls).

For each registered disease the same request is run: fetch (if needed) -> cohort QC ->
signature from raw data -> Hallmark pathways -> ranking with the registered cell line ->
reference-drug audit -> report; literature review is excluded so no model is called
(it is covered by the LUAD frozen review and the live breast-cancer run). Reference-drug
lists were committed before any ranking for the three new diseases was computed.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from drug_repurposing_agent.agent_v2 import RulePlannerV2, run_agent_v2
from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.geo_cohort import files_ready, load_registry
from drug_repurposing_agent.luad_tools_v2 import available_inputs, real_backend
from drug_repurposing_agent.trace import TraceRecorder, traced_run
from drug_repurposing_agent.workflow import Mode

OUTPUT = Path("benchmark/results/registry_validation_v1.json")
ROOT = Path("artifacts/agent_v2_runs/registry_validation")


def _main(trace: TraceRecorder) -> None:
    rows = {}
    for disease_id, entry in load_registry().items():
        question = f"请为{entry['aliases'][0]}筛选候选药物并分析通路，不查文献，出报告"
        on_disk_before = files_ready(entry)
        started = perf_counter()
        report = run_agent_v2(question, Mode.RESEARCH_OPEN, available_inputs(question), RulePlannerV2(),
                              real_backend(ROOT / disease_id, Mode.RESEARCH_OPEN, disease=entry), ROOT / disease_id)
        seconds = round(perf_counter() - started, 1)
        summaries = {e["tool"]: e.get("summary") for e in report["executed"] if e["status"] == "ok"}
        rows[disease_id] = {"label": entry["label"], "accession": entry["accession"], "design": entry.get("design", "paired"),
                            "cell_line": entry["drug_cell_line"], "question": question, "status": report["status"],
                            "files_on_disk_before_run": on_disk_before, "seconds": seconds,
                            "plan": [s["tool"] for r in report["rounds"] for s in (r.get("steps") or [])],
                            "summaries": summaries,
                            "failures": [e for e in report["executed"] if e["status"] == "failed"]}
        trace.emit("disease_validated", disease=disease_id, status=report["status"], seconds=seconds)
        print(disease_id, report["status"], seconds, "s")
    result = {"eval_name": "registry_validation_v1", "note": __doc__, "diseases": rows, "trace_file": str(trace.path)}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(OUTPUT), sha256=sha256_file(OUTPUT))


def main() -> None:
    traced_run("registry_validation_v1", _main)


if __name__ == "__main__":
    main()
