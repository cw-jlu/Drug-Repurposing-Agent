"""Archive the agent v2.1 registry demonstrations (real tools) into one result file.

Runs summarised (made with scripts/run_agent_v2.py on 2026-10-03):
  fetch_demo_rule / fetch_demo_deepseek - GSE32863 raw files moved aside first, so the
      agent had to download them from NCBI, verify SHA-256 and recompute the signature;
  unregistered_demo_deepseek - a request for a disease that is not in the registry.
"""

from __future__ import annotations

import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run

RUNS = Path("artifacts/agent_v2_runs")
NAMES = ("fetch_demo_rule", "fetch_demo_deepseek", "unregistered_demo_deepseek")
OUTPUT = Path("benchmark/results/agent_v2_registry_runs.json")


def _main(trace: TraceRecorder) -> None:
    runs = {}
    for name in NAMES:
        path = RUNS / name / "agent_v2_run.json"
        r = json.loads(path.read_text(encoding="utf-8"))
        runs[name] = {"question": r["question"], "planner": r["planner"], "status": r["status"],
                      "available_inputs": r["available_inputs"],
                      "plans": [[s["tool"] for s in (x.get("steps") or [])] for x in r["rounds"]],
                      "executed": [{k: e[k] for k in ("tool", "status", "summary", "reason") if k in e}
                                   for e in r["executed"]],
                      "source_sha256": sha256_file(path)}
        trace.emit("run_summarised", run=name, status=r["status"])
    result = {"eval_name": "agent_v2_registry_runs", "registry": "configs/disease_registry_v1.json",
              "note": __doc__, "runs": runs, "trace_file": str(trace.path)}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(OUTPUT), sha256=sha256_file(OUTPUT))
    print({k: v["status"] for k, v in runs.items()})


def main() -> None:
    traced_run("agent_v2_registry_runs", _main)


if __name__ == "__main__":
    main()
