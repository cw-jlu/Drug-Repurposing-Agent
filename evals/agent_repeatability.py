"""End-to-end repeatability of the strict-mode TRANSCRIPT Agent run (plan section 10.7).

Each planner (deterministic rule planner, live DeepSeek) runs the same bounded task
10 times: five fixed paraphrases x two repeats. For every run we record the final
status, the validated tool-call plan, wall time, provider usage and the SHA-256 of
the RRF score matrix; matrices are compared with the first completed run by
Spearman correlation. Acceptance gates from the project plan: completion >= 90%
and ranking Spearman >= 0.95. Model-visible traces are written by the Agent under
each run's ignored output directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from drug_repurposing_agent.agent import AgentInputs, Mode, RulePlanner, run_agent_task
from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run

QUESTIONS = (
    "使用 TRANSCRIPT 表达矩阵生成药物重定位排名",
    "请根据转录组数据为所有药物和疾病计算重定位分数",
    "用已有的药物与疾病表达签名做反转排序",
    "Rank drug repurposing candidates from the TRANSCRIPT expression matrices",
    "请对 TRANSCRIPT 数据运行表达反转并输出排名矩阵",
)
DATA = Path("data/raw/TRANSCRIPT_dataset_v2.0.0")


def _rrf(output: Path) -> Path | None:
    found = sorted(output.rglob("rrf.csv"))
    return found[0] if found else None


def summarize(runs: list[dict], reference: pd.DataFrame | None) -> dict:
    done = [r for r in runs if r["status"] == "completed"]
    plans = {json.dumps(r["plan_calls"], sort_keys=True, ensure_ascii=False) for r in done}
    rho = [r["spearman_vs_first"] for r in done if r.get("spearman_vs_first") is not None]
    lat = np.array([r["seconds"] for r in runs])
    return {"runs": len(runs), "completed": len(done), "completion_rate": len(done) / len(runs),
            "distinct_plans_among_completed": len(plans),
            "distinct_rrf_hashes": len({r["rrf_sha256"] for r in done}),
            "min_spearman_vs_first": float(min(rho)) if rho else None,
            "seconds_p50": float(np.percentile(lat, 50)), "seconds_p95": float(np.percentile(lat, 95)),
            "prompt_tokens": int(sum(r.get("usage", {}).get("prompt_tokens", 0) for r in runs)),
            "completion_tokens": int(sum(r.get("usage", {}).get("completion_tokens", 0) for r in runs)),
            "gate_completion_ge_0.90": len(done) / len(runs) >= 0.90,
            "gate_spearman_ge_0.95": bool(rho) and min(rho) >= 0.95}


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--planners", default="rule,deepseek")
    parser.add_argument("--root", type=Path, default=Path("artifacts/agent_repeatability"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/agent_repeatability_v1.json"))
    args = parser.parse_args()
    inputs = AgentInputs(items=DATA / "items.csv", users=DATA / "users.csv")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "mode": "benchmark_strict",
              "questions": list(QUESTIONS), "repeats_per_question": args.repeats,
              "input_sha256": {p.name: sha256_file(p) for p in (inputs.items, inputs.users)},
              "planners": {}, "trace_file": str(trace.path)}
    for name in args.planners.split(","):
        if name == "deepseek":
            from drug_repurposing_agent.deepseek import DeepSeekPlanner
            make = DeepSeekPlanner.from_env
        else:
            make = RulePlanner
        runs, reference = [], None
        for qi, question in enumerate(QUESTIONS):
            for rep in range(args.repeats):
                out = args.root / name / f"q{qi}_r{rep}"
                started = perf_counter()
                try:
                    result = run_agent_task(question, inputs, out, Mode.BENCHMARK_STRICT, make())
                    status, error = str(result.get("status")), None
                except Exception as exc:  # recorded, never hidden
                    result, status, error = {}, "exception", f"{type(exc).__name__}: {exc}"
                seconds = perf_counter() - started
                row = {"question_index": qi, "repeat": rep, "status": status, "seconds": round(seconds, 2),
                       "plan_calls": [c.get("tool") or c.get("name") for c in (result.get("plan") or {}).get("calls", [])],
                       "usage": ((result.get("planner_metadata") or {}).get("usage") or {}), "error": error,
                       "agent_trace": result.get("trace_file")}
                rrf = _rrf(out) if status == "completed" else None
                if rrf is not None:
                    row["rrf_sha256"] = sha256_file(rrf)
                    matrix = pd.read_csv(rrf, index_col=0)
                    if reference is None:
                        reference = matrix
                    row["spearman_vs_first"] = float(spearmanr(reference.to_numpy().ravel(),
                                                               matrix.loc[reference.index, reference.columns].to_numpy().ravel())[0])
                runs.append(row)
                trace.emit("run", planner=name, **{k: v for k, v in row.items() if k != "usage"})
                print(f"{name} q{qi} r{rep}: {status} {seconds:.1f}s plan={row['plan_calls']} rho={row.get('spearman_vs_first')}")
        report["planners"][name] = {"summary": summarize(runs, reference), "runs": runs}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))
    for name, body in report["planners"].items():
        print(name, json.dumps(body["summary"], ensure_ascii=False))


def main() -> None:
    traced_run("agent_repeatability_v1", _main)


if __name__ == "__main__":
    main()
