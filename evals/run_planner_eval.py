"""Run the frozen planner-routing evaluation without executing scientific tools."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from drug_repurposing_agent.agent import PlanningContext, RulePlanner, TOOLS
from drug_repurposing_agent.deepseek import DeepSeekPlanner
from drug_repurposing_agent.workflow import Mode


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(planner, case: dict) -> dict:
    mode = Mode(case["mode"])
    context = PlanningContext(mode, tuple(case["available_inputs"]))
    tools = tuple(spec.public_schema() for spec in TOOLS.values() if mode in spec.modes)
    started = perf_counter()
    try:
        plan = planner.plan(case["question"], context, tools)
        actual = plan.calls[0].name if len(plan.calls) == 1 else "multiple_calls"
        error = None
    except Exception as exc:
        actual = "planner_error"
        error = type(exc).__name__
    elapsed = round((perf_counter() - started) * 1000, 1)
    record = {
        "id": case["id"], "expected_tool": case["expected_tool"],
        "actual_tool": actual, "correct": actual == case["expected_tool"],
        "latency_ms": elapsed, "error_type": error,
    }
    metadata = getattr(planner, "last_metadata", None)
    if isinstance(metadata, dict) and metadata:
        record["provider_metadata"] = metadata
    return record


def evaluate(name: str, planner, cases: list[dict]) -> dict:
    records = [run_one(planner, case) for case in cases]
    usage = [record.get("provider_metadata", {}).get("usage", {}) for record in records]
    return {
        "planner": name,
        "cases": len(records),
        "correct": sum(record["correct"] for record in records),
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "mean_latency_ms": round(mean(record["latency_ms"] for record in records), 1),
        "total_tokens": sum(item.get("total_tokens", 0) for item in usage),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("configs/planner_eval_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner", choices=["rule", "deepseek", "both"], default="both")
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args()
    definition = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = definition["cases"]
    results = []
    if args.planner in {"rule", "both"}:
        results.append(evaluate("rule_fallback_v1", RulePlanner(), cases))
    if args.planner in {"deepseek", "both"}:
        results.append(evaluate(f"deepseek_tool_calling:{args.model}",
                                DeepSeekPlanner.from_env(args.model), cases))
    report = {
        "eval_name": definition["name"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "case_file": str(args.cases),
        "case_file_sha256": sha256(args.cases),
        "contains_raw_model_responses": False,
        "contains_api_key": False,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for result in results:
        print(f"{result['planner']}: {result['correct']}/{result['cases']} "
              f"accuracy={result['accuracy']:.3f} latency_ms={result['mean_latency_ms']} "
              f"tokens={result['total_tokens']}")


if __name__ == "__main__":
    main()
