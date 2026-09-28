"""Stage a separately disclosed positive-only amendment after split infeasibility.

The original five-partition inputs, choices and failed run are never overwritten.
Only explicit -1 ratings become unknown (0); every other cell and feature is kept.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run, verify_trace_chain
from evals.grade_method_selection_traces import grade_choices


RULE = "Map every explicit -1 rating to 0 (unknown); preserve 0, 1 and all features."
FILES = ("ratings_mat.csv", "items.csv", "users.csv")


def positive_only_ratings(ratings: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if not ratings.isin((-1, 0, 1)).all().all():
        raise ValueError("Unexpected rating value")
    negatives = int((ratings == -1).sum().sum())
    return ratings.mask(ratings == -1, 0), negatives


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path,
                        default=Path("configs/method_selection_partition_v2_cases.json"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("configs/method_selection_partition_v2.json"))
    parser.add_argument("--choices", type=Path,
                        default=Path("artifacts/reports/method_selection_partition_v2_choices.json"))
    parser.add_argument("--failed-run-trace", type=Path,
                        default=Path("artifacts/traces/20260926T153328294433Z_method_selection_partition_v2_benchmarks_0b04ca0f58ec49a7abdc38accf0a9e8a.jsonl"))
    parser.add_argument("--root", type=Path,
                        default=Path("artifacts/method_selection_partition_v2_positive_only"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/reports/method_selection_partition_v2_positive_only_manifest.json"))
    parser.add_argument("--verify-existing-manifest", action="store_true",
                        help="Rebuild missing staged data and verify the committed manifest without rewriting it")
    args = parser.parse_args()
    trace.emit("amendment_staging_started", rule=RULE, cases=str(args.cases),
               failed_run_trace=str(args.failed_run_trace))
    if args.verify_existing_manifest:
        existing_manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if (existing_manifest.get("status") !=
                "post_failure_feasibility_amendment_not_original_protocol" or
                existing_manifest.get("rule") != RULE or
                existing_manifest.get("protocol_sha256") != sha256_file(args.protocol) or
                existing_manifest.get("cases_sha256") != sha256_file(args.cases) or
                existing_manifest.get("choices_sha256") != sha256_file(args.choices) or
                existing_manifest.get("failed_run_trace_sha256") != sha256_file(args.failed_run_trace)):
            raise ValueError("Committed amendment manifest differs from frozen inputs")
    elif args.manifest.exists() or args.root.exists():
        raise ValueError("Amendment output already exists; refusing to overwrite")
    grade = grade_choices(args.choices, args.cases, args.protocol)
    if grade["choices"] != 15 or grade["passed"] != 15:
        raise ValueError("All frozen choices must pass trace grading")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    failed_events, _ = verify_trace_chain(args.failed_run_trace, require_chain=True)
    if (failed_events[-1].get("stage") != "run_failed" or
            not any(event.get("stage") == "choices_locked" for event in failed_events)):
        raise ValueError("Original failure trace does not establish locked choices and failure")
    entries = []
    for case in cases:
        original = Path(case["dataset_dir"])
        for name in FILES:
            if sha256_file(original / name) != case["staged_sha256"][name]:
                raise ValueError(f"Original staged data changed: {case['id']}/{name}")
        ratings = pd.read_csv(original / "ratings_mat.csv", index_col=0)
        amended, negative_count = positive_only_ratings(ratings)
        destination = args.root / case["id"] / "datasets" / "TRANSCRIPT"
        if destination.exists():
            if not args.verify_existing_manifest:
                raise ValueError(f"Amended dataset already exists: {destination}")
        else:
            destination.mkdir(parents=True, exist_ok=False)
            amended.to_csv(destination / "ratings_mat.csv")
            for name in ("items.csv", "users.csv"):
                shutil.copy2(original / name, destination / name)
        hashes = {name: sha256_file(destination / name) for name in FILES}
        for name in ("items.csv", "users.csv"):
            if hashes[name] != case["staged_sha256"][name]:
                raise ValueError(f"Feature file changed during staging: {case['id']}/{name}")
        if not pd.read_csv(destination / "ratings_mat.csv", index_col=0).equals(amended):
            raise ValueError(f"Amended ratings did not round-trip: {case['id']}")
        entry = {"case_id": case["id"], "dataset_dir": str(destination),
                 "original_negative_count": negative_count,
                 "original_sha256": case["staged_sha256"], "amended_sha256": hashes}
        entries.append(entry)
        trace.emit("partition_staged", **entry)
    manifest = {"status": "post_failure_feasibility_amendment_not_original_protocol",
                "created_at": datetime.now(timezone.utc).isoformat(), "rule": RULE,
                "protocol_sha256": sha256_file(args.protocol),
                "cases_sha256": sha256_file(args.cases),
                "choices_sha256": sha256_file(args.choices),
                "failed_run_trace_sha256": sha256_file(args.failed_run_trace),
                "partitions": entries, "trace_file": str(trace.path),
                "disclosure": ("The original preregistered five-partition run failed at partition 02 "
                               "because one explicit negative cannot be stratified. This mapping "
                               "is a post-failure protocol amendment, not a clean confirmatory run.")}
    if args.verify_existing_manifest:
        if existing_manifest["partitions"] != entries:
            raise ValueError("Rebuilt positive-only inputs differ from committed manifest")
        trace.emit("amendment_staging_verified", manifest=str(args.manifest),
                   manifest_sha256=sha256_file(args.manifest), partitions=len(entries))
        print(f"Verified {len(entries)} positive-only partitions: {args.manifest}")
        return
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_bytes(json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
    trace.emit("amendment_staging_completed", manifest=str(args.manifest),
               manifest_sha256=sha256_file(args.manifest),
               negative_counts={entry["case_id"]: entry["original_negative_count"]
                                for entry in entries})
    print(f"Staged {len(entries)} positive-only partitions: {args.manifest}")


def main() -> None:
    traced_run("method_selection_positive_only_staging", _main)


if __name__ == "__main__":
    main()
