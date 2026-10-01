"""Run and grade the frozen multi-step planning eval (agent v2, simulated tools).

The case file hash (LF-normalised) is pinned below and checked before any call.
Each case runs the full agent v2 loop (plan -> validate -> execute -> observe ->
re-plan) against the simulated backend with the case's injected failures, then is
graded by grade(). Planners: rule (deterministic) and DeepSeek submit_plan
(deepseek-flash by default; --model for others). One run per planner.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from drug_repurposing_agent.agent_v2 import DeepSeekPlannerV2, RulePlannerV2, run_agent_v2, simulated_backend
from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run
from drug_repurposing_agent.workflow import Mode

CASES = Path("configs/planner_eval_multistep_v1.json")
FROZEN_SHA256 = "8654f8a4060b314713129ab2439b8921d3baf4015de83cd5ac2eaf6a364686cd"


def load_cases() -> list[dict]:
    digest = hashlib.sha256(CASES.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != FROZEN_SHA256:
        raise ValueError(f"Frozen case file changed: {digest}")
    return json.loads(CASES.read_text(encoding="utf-8"))["cases"]


def grade(case: dict, report: dict) -> dict:
    exp = case["expected"]
    attempted = [e["tool"] for e in report["executed"] if e["status"] in ("ok", "failed")]
    ok = [e["tool"] for e in report["executed"] if e["status"] == "ok"]
    first_ok = {}
    for i, tool in enumerate(ok):
        first_ok.setdefault(tool, i)
    problems = []
    if report["status"] not in exp["status"]:
        problems.append(f"status {report['status']} not in {exp['status']}")
    problems += [f"missing {t}" for t in exp["must_succeed"] if t not in first_ok]
    problems += [f"called forbidden {t}" for t in exp["must_not_call"] if t in attempted]
    for a, b in exp["order"]:
        if a in first_ok and b in first_ok and first_ok[a] > first_ok[b]:
            problems.append(f"order {a} after {b}")
    return {"passed": not problems, "problems": problems, "status": report["status"],
            "rounds": len(report["rounds"]), "tools_ok": ok, "tools_attempted": attempted,
            "invalid_plans": sum(1 for r in report["rounds"] if r.get("invalid_reason")),
            "planner_errors": sum(1 for r in report["rounds"] if r.get("error"))}


def run_case(case: dict, make_planner, root: Path) -> dict:
    planner = make_planner()
    out = root / case["id"]
    report = run_agent_v2(case["question"], Mode(case["mode"]), tuple(case["available_inputs"]), planner,
                          simulated_backend(case["failures"]), out)
    usage = Counter()
    for r in report["rounds"]:
        for k, v in ((r.get("planner_metadata") or {}).get("usage") or {}).items():
            usage[k] += v
    return {"id": case["id"], "category": case["category"], **grade(case, report),
            "plans": [r.get("steps") for r in report["rounds"]], "usage": dict(usage)}


def summarize(rows: list[dict]) -> dict:
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    failure_rows = [r for r in rows if r["category"] == "failure"]
    return {"passed": sum(r["passed"] for r in rows), "cases": len(rows),
            "by_category": {k: f"{sum(v)}/{len(v)}" for k, v in by_cat.items()},
            "mean_rounds": round(sum(r["rounds"] for r in rows) / len(rows), 2),
            "replanned_cases": sum(r["rounds"] > 1 for r in rows),
            "invalid_plans_total": sum(r["invalid_plans"] for r in rows),
            "planner_errors_total": sum(r["planner_errors"] for r in rows),
            "forbidden_calls": sum(any(p.startswith("called forbidden") for p in r["problems"]) for r in rows),
            "failure_recovery": f"{sum(r['passed'] for r in failure_rows)}/{len(failure_rows)}",
            "tokens": dict(sum((Counter(r["usage"]) for r in rows), Counter()))}


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planners", default="rule,deepseek")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/planner_eval_multistep_v1.json"))
    args = parser.parse_args()
    cases = load_cases()
    trace.emit("frozen_cases_verified", sha256=FROZEN_SHA256, cases=len(cases))
    result = {"eval_name": "planner_eval_multistep_v1", "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "case_file": str(CASES), "case_file_sha256_lf": FROZEN_SHA256, "backend": "simulated",
              "contains_api_key": False, "planners": {}, "trace_file": str(trace.path)}
    root = Path(tempfile.mkdtemp(prefix="agent_v2_eval_", dir="artifacts"))
    for name in args.planners.split(","):
        if name == "rule":
            make, label, workers = RulePlannerV2, "rule_v2", 1
        else:
            make, label, workers = (lambda: DeepSeekPlannerV2.from_env(args.model)), f"deepseek_v2:{args.model}", 4
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda c: run_case(c, make, root / label.replace(":", "_")), cases))
        result["planners"][label] = {"summary": summarize(rows), "cases": rows}
        trace.emit("planner_done", planner=label, summary=result["planners"][label]["summary"])
        s = result["planners"][label]["summary"]
        print(label, f"{s['passed']}/{s['cases']}", s["by_category"], "replanned", s["replanned_cases"],
              "forbidden", s["forbidden_calls"])
    result["run_directory"] = str(root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))


def main() -> None:
    traced_run("planner_eval_multistep_v1", _main)


if __name__ == "__main__":
    main()
