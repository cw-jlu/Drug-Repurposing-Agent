"""Run the traced multi-agent literature triage over the frozen LUAD Top-10.

Literature triage only: tiers describe retrievable PubMed evidence, not efficacy.
Per-candidate results are checkpointed so a rerun resumes without new model calls.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.deepseek import local_api_key
from drug_repurposing_agent.llm_calls import CallStats, TracedToolCaller, estimate_cost_usd
from drug_repurposing_agent.multi_agent_review import (AbstractPubMedClient, MultiAgentReviewer,
                                                       TIERS)
from drug_repurposing_agent.trace import TraceRecorder, traced_run

EXISTING_TIER_MAP = {"insufficient_evidence": "INSUFFICIENT_EVIDENCE"}


def summarize(rows: list[dict], evidence: dict, stats: dict, model: str, hashes: dict,
              trace_path: str, checkpoint_sha: str | None) -> dict:
    existing = {c["rank"]: c["confidence"] for c in evidence["candidates"]}
    proposed = sum(r["proposed_claims"]["support"] + r["proposed_claims"]["contradiction"]
                   for r in rows)
    rejected = sum(r["rejected_claims"]["support"] + r["rejected_claims"]["contradiction"]
                   for r in rows)
    reasons: dict[str, int] = {}
    for row in rows:
        for item in row["rejected_claims"]["details"]:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    comparison = []
    for row in rows:
        single = existing[row["rank"]]
        comparison.append({"rank": row["rank"], "name": row["name"],
                           "single_pass_confidence": single, "multi_agent_tier": row["tier"],
                           "agrees": EXISTING_TIER_MAP.get(single) == row["tier"]})
    return {
        "experiment": "multi_agent_review_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "literature_triage_only_not_efficacy_evidence",
        "model": model,
        "tiers_allowed": list(TIERS),
        "input_sha256": hashes,
        "tier_counts": {tier: sum(r["tier"] == tier for r in rows) for tier in TIERS},
        "candidates": rows,
        "comparison_with_single_pass": comparison,
        "citation_validation": {
            "proposed_quoted_items": proposed, "rejected_quoted_items": rejected,
            "rejection_rate": round(rejected / proposed, 4) if proposed else None,
            "rejection_reasons": reasons,
            "rule": ("PMID must be in the agent's retrieved set and the quote must be a "
                     "whitespace-normalized verbatim substring (>=20 chars) of that abstract"),
        },
        "call_counts": stats,
        "cost_estimate": estimate_cost_usd(stats.get("usage", {})),
        "trace_file": trace_path,
        "checkpoint_sha256": checkpoint_sha,
        "interpretation": ("Tiers triage retrievable literature for follow-up research. They are "
                           "not treatment recommendations and do not establish LUAD efficacy."),
    }


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=Path("configs/luad_top10_evidence_v1.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/results/multi_agent_review_v1.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/multi_agent_review"))
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    parser.add_argument("--max-calls", type=int, default=60)
    args = parser.parse_args()
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DRUG_AGENT_TRACE_DIR", str(work / "provider_traces"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    hashes = {"evidence": sha256_file(args.evidence)}
    trace.emit("input_loaded", evidence=str(args.evidence), input_sha256=hashes,
               candidates=[c["name"] for c in evidence["candidates"]])
    checkpoint = work / "checkpoint_v1.jsonl"
    stats_path = work / "call_stats.json"
    done: dict[int, dict] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["evidence_sha256"] == hashes["evidence"]:
                    done[row["result"]["rank"]] = row["result"]
    previous = json.loads(stats_path.read_text()) if stats_path.exists() else None
    key = local_api_key()
    trace.add_secret(key)
    stats = CallStats()
    caller = TracedToolCaller(key, model=args.model, stats=stats)
    reviewer = MultiAgentReviewer(caller, AbstractPubMedClient(), trace, model=args.model)
    for candidate in evidence["candidates"]:
        if candidate["rank"] in done:
            continue
        if stats.as_dict()["attempts"] + 3 > args.max_calls:
            raise RuntimeError("Call budget would be exceeded")
        trace.emit("candidate_started", rank=candidate["rank"], name=candidate["name"])
        result = reviewer.review(candidate)
        done[candidate["rank"]] = result
        with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"evidence_sha256": hashes["evidence"], "result": result},
                                    ensure_ascii=False) + "\n")
        print(f"{candidate['rank']:>2} {candidate['name']}: {result['tier']}", flush=True)
    current = stats.as_dict()
    if previous:
        current = {"attempts": current["attempts"] + previous["attempts"],
                   "successes": current["successes"] + previous["successes"],
                   "failures": current["failures"] + previous["failures"],
                   "usage": {k: current["usage"].get(k, 0) + previous["usage"].get(k, 0)
                             for k in set(current["usage"]) | set(previous["usage"])}}
    stats_path.write_text(json.dumps(current, indent=1), encoding="utf-8")
    rows = [done[rank] for rank in sorted(done)]
    result = summarize(rows, evidence, current, args.model, hashes, str(trace.path),
                       sha256_file(checkpoint) if checkpoint.exists() else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trace.emit("results_saved", output=str(args.output), output_sha256=sha256_file(args.output),
               tier_counts=result["tier_counts"], citation_validation=result["citation_validation"])
    print(json.dumps({"tier_counts": result["tier_counts"],
                      "citation_validation": result["citation_validation"],
                      "calls": current}, indent=1))


if __name__ == "__main__":
    traced_run("multi_agent_review", _main, Path("artifacts/multi_agent_review/traces"))
