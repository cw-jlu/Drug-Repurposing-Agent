"""Check external disease-ID overlap with the already analyzed TRANSCRIPT dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


AUDIT = Path("benchmark/results/external_repodb_exact_crosswalk_audit.json")
TRANSCRIPT = Path("data/raw/TRANSCRIPT_dataset_v2.0.0/users.csv")
OUTPUT = Path("benchmark/results/external_repodb_independence_audit.json")


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    rows = json.loads(AUDIT.read_text(encoding="utf-8"))["rows"]
    development_ids = set(pd.read_csv(TRANSCRIPT, nrows=0).columns[1:])
    matched = [row for row in rows if row["indication_id"]]
    overlap = [row for row in matched if row["indication_id"] in development_ids]
    fresh = [row for row in matched if row["indication_id"] not in development_ids]
    both = [row for row in fresh if row["approved_drugs"] and row["failed_trial_drugs"]]
    result = {"status": "source_overlap_audit_only_not_external_benchmark",
              "repodb_crosswalk_sha256": sha256_file(AUDIT),
              "transcript_users_sha256": sha256_file(TRANSCRIPT),
              "transcript_disease_ids": len(development_ids),
              "mapped_external_disease_ids": len(matched),
              "identical_cui_overlap_count": len(overlap),
              "nonoverlap_disease_ids": len(fresh),
              "nonoverlap_approved_pairs": sum(row["approved_drugs"] for row in fresh),
              "nonoverlap_failed_trial_pairs": sum(row["failed_trial_drugs"] for row in fresh),
              "nonoverlap_diseases_with_both_labels": len(both),
              "nonoverlap_max_positive_negative_counts": sorted(
                  [{"source_name": row["source_name"], "positive": row["approved_drugs"],
                    "negative": row["failed_trial_drugs"]} for row in both],
                  key=lambda row: min(row["positive"], row["negative"]), reverse=True),
              "note": "Different CUI does not prove independent underlying CREEDS GEO cohorts; that requires source-signature provenance matching.",
              "trace": str(trace.path)}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("audit_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               overlap=len(overlap), fresh=len(fresh),
               fresh_positive=result["nonoverlap_approved_pairs"],
               fresh_negative=result["nonoverlap_failed_trial_pairs"])
    print(json.dumps({key: result[key] for key in (
        "identical_cui_overlap_count", "nonoverlap_disease_ids",
        "nonoverlap_approved_pairs", "nonoverlap_failed_trial_pairs",
        "nonoverlap_diseases_with_both_labels")}, indent=2))
    print(f"Trace: {trace.path}")


if __name__ == "__main__":
    traced_run("repodb_independence_audit", _main, Path("artifacts/reports/traces"))
