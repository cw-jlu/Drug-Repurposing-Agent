"""Audit explicit repoDB labels against staged CDRPipe/CMap signatures, no predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
import unicodedata

import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


SOURCE = Path("artifacts/external/repodb_2017/shiny.RData")
ROOT = Path("artifacts/external/cdrpipe-comparative-analysis")
SIGNATURES = ROOT / "creeds/data/manual_signatures_extracted"
CMAP = ROOT / "drug_signatures/data/cmap/cmap_drug_experiments_new.csv"
OUTPUT = Path("benchmark/results/external_repodb_exact_crosswalk_audit.json")
NEGATIVE = {"Terminated", "Withdrawn", "Suspended"}


def _normal(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().replace("_", " ")
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    trace.emit("input_verified", repodb_sha256=sha256_file(SOURCE),
               cmap_metadata_sha256=sha256_file(CMAP),
               disease_signature_count=len(list(SIGNATURES.glob("*_signature.csv"))))
    sys.path.insert(0, str(Path("artifacts/tools/python_packages").resolve()))
    import rdata
    repo = rdata.read_rda(SOURCE, default_encoding="utf-8",
                          force_default_encoding=True)["drug.fr"]
    cmap = pd.read_csv(CMAP)
    if set(repo["status"].dropna().unique()) != {"Approved", *NEGATIVE}:
        raise ValueError("repoDB status domain changed")
    valid_cmap = cmap[cmap["DrugBank.ID"].astype(str).str.fullmatch(r"DB\d{5}")].copy()
    by_id = valid_cmap.groupby("DrugBank.ID")["name"].agg(lambda values: set(values))
    unique_id_to_name = {key: next(iter(names)) for key, names in by_id.items()
                         if len(names) == 1}
    disease_names: dict[str, set[str]] = defaultdict(set)
    for name, identifier in zip(repo["ind_name"], repo["ind_id"]):
        if pd.notna(name) and pd.notna(identifier):
            disease_names[_normal(name)].add(str(identifier))
    labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for disease_id, drug_id, status in zip(repo["ind_id"], repo["drug_id"], repo["status"]):
        if pd.notna(disease_id) and drug_id in unique_id_to_name:
            labels[(str(disease_id), drug_id)].add(status)

    rows = []
    for path in sorted(SIGNATURES.glob("*_signature.csv")):
        name = path.name.removesuffix("_signature.csv").replace("_", " ")
        ids = disease_names.get(_normal(name), set())
        disease_id = next(iter(ids)) if len(ids) == 1 else None
        positive = negative = conflicting = 0
        negative_reasons = Counter()
        if disease_id:
            for (indication, _drug), statuses in labels.items():
                if indication != disease_id:
                    continue
                if "Approved" in statuses and statuses & NEGATIVE:
                    conflicting += 1
                elif "Approved" in statuses:
                    positive += 1
                elif statuses & NEGATIVE:
                    negative += 1
                    negative_reasons.update(statuses)
        rows.append({"signature": path.name, "source_name": name,
                     "indication_id": disease_id,
                     "match_status": "unique_exact_name" if disease_id else
                     ("ambiguous_name" if ids else "unmatched"),
                     "approved_drugs": positive, "failed_trial_drugs": negative,
                     "conflicting_pairs_excluded": conflicting,
                     "failed_status_counts": dict(negative_reasons),
                     "five_fold_eligible_min_8_each": positive >= 8 and negative >= 8})
    result = {
        "status": "crosswalk_feasibility_only_no_method_predictions",
        "source_hashes": {str(SOURCE): sha256_file(SOURCE), str(CMAP): sha256_file(CMAP)},
        "label_policy_audited_not_yet_benchmark_frozen": (
            "Approved=positive; Terminated/Withdrawn/Suspended=failed-trial negative; "
            "same-pair conflicting statuses excluded; unmatched/unknown never labeled negative. "
            "These trial statuses are not necessarily lack of efficacy."),
        "crosswalk_policy": "unique normalized exact disease name to UMLS CUI, unique DrugBank ID to one CMap name; no fuzzy/entity inference",
        "drugbank_ids_in_cmap": int(cmap["DrugBank.ID"].nunique(dropna=True)),
        "unique_unambiguous_drugbank_ids": len(unique_id_to_name),
        "disease_match_counts": dict(Counter(row["match_status"] for row in rows)),
        "total_approved_pairs": sum(row["approved_drugs"] for row in rows),
        "total_failed_trial_pairs": sum(row["failed_trial_drugs"] for row in rows),
        "total_conflicting_pairs_excluded": sum(row["conflicting_pairs_excluded"] for row in rows),
        "diseases_with_both_labels": sum(row["approved_drugs"] > 0 and
                                         row["failed_trial_drugs"] > 0 for row in rows),
        "five_fold_eligible_disease_units_min_8_each": sum(
            row["five_fold_eligible_min_8_each"] for row in rows),
        "rows": rows, "trace": str(trace.path),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("audit_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               matched_diseases=result["disease_match_counts"],
               approved=result["total_approved_pairs"],
               failed=result["total_failed_trial_pairs"],
               eligible=result["five_fold_eligible_disease_units_min_8_each"])
    print(json.dumps({key: result[key] for key in (
        "unique_unambiguous_drugbank_ids", "disease_match_counts", "total_approved_pairs",
        "total_failed_trial_pairs", "total_conflicting_pairs_excluded",
        "diseases_with_both_labels", "five_fold_eligible_disease_units_min_8_each")},
        ensure_ascii=True, indent=2))
    print(f"Trace: {trace.path}")


if __name__ == "__main__":
    traced_run("repodb_exact_crosswalk_audit", _main, Path("artifacts/reports/traces"))
