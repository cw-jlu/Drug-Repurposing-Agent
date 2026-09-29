"""Test a conservative Open Targets UMLS bridge for repoDB label feasibility."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


BRIDGE = Path("benchmark/results/external_ot_repodb_bridge_audit.json")
SOURCE = Path("artifacts/external/repodb_2017/shiny.RData")
CMAP = Path("artifacts/external/cdrpipe-comparative-analysis/drug_signatures/data/cmap/cmap_drug_experiments_new.csv")
TRANSCRIPT = Path("data/raw/TRANSCRIPT_dataset_v2.0.0/users.csv")
OUTPUT = Path("benchmark/results/external_repodb_ontology_bridge_audit.json")
NEGATIVE = {"Terminated", "Withdrawn", "Suspended"}


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    bridge = json.loads(BRIDGE.read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path("artifacts/tools/python_packages").resolve()))
    import rdata
    repo = rdata.read_rda(SOURCE, default_encoding="utf-8",
                          force_default_encoding=True)["drug.fr"]
    cmap = pd.read_csv(CMAP)
    development_ids = set(pd.read_csv(TRANSCRIPT, nrows=0).columns[1:])
    drug_names = cmap[cmap["DrugBank.ID"].astype(str).str.fullmatch(r"DB\d{5}")].groupby(
        "DrugBank.ID")["name"].agg(lambda values: set(values))
    mapped_drugs = {identifier for identifier, names in drug_names.items() if len(names) == 1}
    status_by_pair: dict[tuple[str, str], set[str]] = defaultdict(set)
    for disease, drug, status in zip(repo["ind_id"], repo["drug_id"], repo["status"]):
        if pd.notna(disease) and drug in mapped_drugs:
            status_by_pair[(str(disease), drug)].add(status)
    known_cuis = set(repo["ind_id"].dropna().astype(str))
    rows = []
    for route in bridge["routes"]:
        candidates = set(route["ot_cuis"]) & known_cuis
        if route["repodb_exact_cui"]:
            candidates.add(route["repodb_exact_cui"])
        if len(candidates) != 1:
            continue
        cui = next(iter(candidates))
        positive = negative = conflicting = 0
        for (disease, _drug), statuses in status_by_pair.items():
            if disease != cui:
                continue
            if "Approved" in statuses and statuses & NEGATIVE:
                conflicting += 1
            elif "Approved" in statuses:
                positive += 1
            elif statuses & NEGATIVE:
                negative += 1
        rows.append({"signature": route["signature"], "cui": cui,
                     "cui_in_transcript": cui in development_ids,
                     "positive": positive, "failed_trial_negative": negative,
                     "conflicting_excluded": conflicting,
                     "five_fold_eligible_min_8_each": positive >= 8 and negative >= 8})
    fresh = [row for row in rows if not row["cui_in_transcript"]]
    result = {"status": "conservative_ontology_bridge_feasibility_only_no_predictions",
              "inputs_sha256": {str(path): sha256_file(path) for path in
                                (BRIDGE, SOURCE, CMAP, TRANSCRIPT)},
              "crosswalk": "Unique UMLS CUI among Open Targets dbXRefs that occurs in repoDB; exact-name CUI may confirm; ambiguous CUIs excluded; unambiguous CMap DrugBank IDs only",
              "mapped_signatures": len(rows),
              "mapped_positives": sum(row["positive"] for row in rows),
              "mapped_failed_trial_negatives": sum(row["failed_trial_negative"] for row in rows),
              "mapped_with_both_labels": sum(bool(row["positive"] and row["failed_trial_negative"]) for row in rows),
              "mapped_five_fold_eligible_min_8_each": sum(row["five_fold_eligible_min_8_each"] for row in rows),
              "development_cui_overlap": len(rows) - len(fresh),
              "nonoverlap_signatures": len(fresh),
              "nonoverlap_positives": sum(row["positive"] for row in fresh),
              "nonoverlap_failed_trial_negatives": sum(row["failed_trial_negative"] for row in fresh),
              "nonoverlap_with_both_labels": sum(bool(row["positive"] and row["failed_trial_negative"]) for row in fresh),
              "nonoverlap_five_fold_eligible_min_8_each": sum(row["five_fold_eligible_min_8_each"] for row in fresh),
              "rows": rows, "trace": str(trace.path)}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("audit_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               mapped=result["mapped_signatures"],
               nonoverlap=result["nonoverlap_signatures"],
               eligible=result["nonoverlap_five_fold_eligible_min_8_each"])
    print(json.dumps({key: result[key] for key in (
        "mapped_signatures", "mapped_positives", "mapped_failed_trial_negatives",
        "mapped_with_both_labels", "mapped_five_fold_eligible_min_8_each",
        "development_cui_overlap", "nonoverlap_signatures", "nonoverlap_positives",
        "nonoverlap_failed_trial_negatives", "nonoverlap_with_both_labels",
        "nonoverlap_five_fold_eligible_min_8_each")}, indent=2))
    print(f"Trace: {trace.path}")


if __name__ == "__main__":
    traced_run("repodb_ontology_bridge_audit", _main, Path("artifacts/reports/traces"))
