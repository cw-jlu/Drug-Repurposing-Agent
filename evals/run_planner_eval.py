"""Run the frozen planner-routing evaluation without executing scientific tools."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from drug_repurposing_agent.agent import (
    PlanningContext,
    RulePlanner,
    TOOLS,
    validate_tool_call,
)
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
        if len(plan.calls) != 1:
            raise ValueError("Planner must return exactly one tool call")
        validate_tool_call(plan.calls[0], mode)
        actual = plan.calls[0].name
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


def estimate_cost(records: list[dict], pricing_period: str) -> dict | None:
    if pricing_period == "none":
        return None
    prices = {
        "offpeak": {"cache_hit": 0.003, "cache_miss": 0.15, "output": 0.6},
        "peak": {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2},
    }[pricing_period]
    usage = [record.get("provider_metadata", {}).get("usage", {}) for record in records]
    prompt = sum(item.get("prompt_tokens", 0) for item in usage)
    hit = sum(item.get("prompt_cache_hit_tokens", 0) for item in usage)
    miss_reported = sum(item.get("prompt_cache_miss_tokens", 0) for item in usage)
    miss = miss_reported if hit + miss_reported == prompt else prompt - hit
    output = sum(item.get("completion_tokens", 0) for item in usage)
    cost = (hit * prices["cache_hit"] + miss * prices["cache_miss"] +
            output * prices["output"]) / 1_000_000
    return {
        "period": pricing_period,
        "pricing_checked_on": "2026-09-24",
        "currency": "USD",
        "per_million_tokens": prices,
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": miss,
        "completion_tokens": output,
        "estimated_cost_usd": round(cost, 6),
    }


def evaluate(name: str, planner, cases: list[dict], pricing_period: str) -> dict:
    records = [run_one(planner, case) for case in cases]
    usage = [record.get("provider_metadata", {}).get("usage", {}) for record in records]
    result = {
        "planner": name,
        "cases": len(records),
        "correct": sum(record["correct"] for record in records),
        "accuracy": sum(record["correct"] for record in records) / len(records),
        "mean_latency_ms": round(mean(record["latency_ms"] for record in records), 1),
        "total_tokens": sum(item.get("total_tokens", 0) for item in usage),
        "records": records,
    }
    cost = estimate_cost(records, pricing_period)
    if cost is not None and name.startswith("deepseek_"):
        result["cost_estimate"] = cost
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("configs/planner_eval_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planner", choices=["rule", "deepseek", "both"], default="both")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--pricing-period", choices=["none", "offpeak", "peak"], default="none")
    args = parser.parse_args()
    definition = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = definition["cases"]
    results = []
    if args.planner in {"rule", "both"}:
        results.append(evaluate("rule_fallback_v1", RulePlanner(), cases, "none"))
    if args.planner in {"deepseek", "both"}:
        results.append(evaluate(f"deepseek_tool_calling:{args.model}",
                                DeepSeekPlanner.from_env(args.model), cases,
                                args.pricing_period))
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
        cost = result.get("cost_estimate", {}).get("estimated_cost_usd", 0)
        print(f"{result['planner']}: {result['correct']}/{result['cases']} "
              f"accuracy={result['accuracy']:.3f} latency_ms={result['mean_latency_ms']} "
              f"tokens={result['total_tokens']} cost_usd={cost}")


if __name__ == "__main__":
    main()
