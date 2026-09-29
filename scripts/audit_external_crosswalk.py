"""Audit exact external disease/drug name alignment without scoring predictions."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


ROOT = Path("artifacts/external/cdrpipe-comparative-analysis")
SIGNATURES = ROOT / "creeds/data/manual_signatures_extracted"
CMAP_META = ROOT / "drug_signatures/data/cmap/cmap_drug_experiments_new.csv"
OT = Path("artifacts/external/opentargets_26_06")
OUTPUT = Path("benchmark/results/external_exact_crosswalk_audit.json")


def _normal(name: str) -> str:
    name = unicodedata.normalize("NFKC", str(name)).casefold().replace("_", " ")
    return " ".join(re.findall(r"[\w]+", name, flags=re.UNICODE))


def _unique_index(names: list[tuple[str, str]]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for identifier, name in names:
        if isinstance(name, str) and name.strip():
            mapping[_normal(name)].add(identifier)
    return mapping


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    sources = [OT / f"{name}.parquet" for name in
               ("clinical_indication", "disease", "drug_molecule")]
    sources += [CMAP_META]
    if not all(path.is_file() for path in sources):
        raise FileNotFoundError("The three pinned Open Targets tables and CMap metadata must be staged")
    trace.emit("inputs_verified", inputs=[{"path": str(path), "sha256": sha256_file(path)}
                                           for path in sources])
    clinical = pd.read_parquet(sources[0], columns=["drugId", "diseaseId", "maxClinicalStage"])
    disease = pd.read_parquet(sources[1], columns=["id", "name", "exactSynonyms"])
    molecule = pd.read_parquet(sources[2])
    cmap = pd.read_csv(CMAP_META)
    name_field = next((name for name in ("prefName", "name") if name in molecule), None)
    if name_field is None:
        raise ValueError(f"No preferred drug name field; schema: {list(molecule.columns)}")
    trace.emit("schemas_read", clinical_rows=len(clinical), disease_rows=len(disease),
               molecule_rows=len(molecule), cmap_rows=len(cmap),
               molecule_columns=list(molecule.columns), molecule_name_field=name_field)

    disease_names = _unique_index(list(zip(disease["id"], disease["name"])))
    disease_exact_synonyms = _unique_index([
        (identifier, synonym)
        for identifier, terms in zip(disease["id"], disease["exactSynonyms"])
        if pd.api.types.is_list_like(terms) and not isinstance(terms, str)
        for synonym in terms
    ])
    clinical_ids = set(clinical["diseaseId"].dropna())
    disease_rows = []
    for path in sorted(SIGNATURES.glob("*_signature.csv")):
        name = path.name.removesuffix("_signature.csv").replace("_", " ")
        key = _normal(name)
        candidates = disease_names.get(key, set())
        route = "preferred_name"
        if not candidates:
            candidates = disease_exact_synonyms.get(key, set())
            route = "exact_synonym" if candidates else "unmatched"
        if len(candidates) > 1:
            route = "ambiguous"
        identifier = next(iter(candidates)) if len(candidates) == 1 else None
        disease_rows.append({"signature": path.name, "source_name": name,
                             "route": route, "disease_id": identifier,
                             "clinical_indication_present": identifier in clinical_ids
                             if identifier is not None else False,
                             "candidate_ids": sorted(candidates) if identifier is None else []})

    drug_names = _unique_index(list(zip(molecule["id"], molecule[name_field])))
    cmap_drugs = cmap[["name", "DrugBank.ID"]].drop_duplicates()
    drug_rows = []
    for _, item in cmap_drugs.iterrows():
        drug_name = str(item["name"])
        ids = drug_names.get(_normal(drug_name), set())
        identifier = next(iter(ids)) if len(ids) == 1 else None
        drug_rows.append({"cmap_name": drug_name, "drugbank_id": item["DrugBank.ID"]
                          if pd.notna(item["DrugBank.ID"]) else None,
                          "chembl_id": identifier,
                          "route": "preferred_name" if identifier else
                          ("ambiguous" if ids else "unmatched"),
                          "candidate_ids": sorted(ids) if identifier is None else [],
                          "experiment_rows": int((cmap["name"] == item["name"]).sum())})

    disease_counts = Counter(row["route"] for row in disease_rows)
    drug_counts = Counter(row["route"] for row in drug_rows)
    mapped_diseases = {row["disease_id"] for row in disease_rows if row["disease_id"]}
    mapped_drugs = {row["chembl_id"] for row in drug_rows if row["chembl_id"]}
    known_pairs = clinical[clinical["diseaseId"].isin(mapped_diseases) &
                           clinical["drugId"].isin(mapped_drugs)]
    pair_counts = known_pairs.groupby("diseaseId")["drugId"].nunique().to_dict()
    for row in disease_rows:
        row["known_positive_drugs_in_exact_cmap_crosswalk"] = int(
            pair_counts.get(row["disease_id"], 0))
    result = {
        "status": "alignment_audit_only_not_a_negative_labeled_benchmark",
        "normalization": "Unicode NFKC, casefold, underscores/punctuation to spaces; unique preferred names then unique exact disease synonyms; no fuzzy matches",
        "clinical_label_semantics": "Open Targets clinical indications are documented development links, not explicit failures or negatives; absent link is unlabeled",
        "source_hashes": {str(path): sha256_file(path) for path in sources},
        "disease_route_counts": dict(disease_counts), "drug_route_counts": dict(drug_counts),
        "disease_signatures_with_mapped_positive": sum(
            row["known_positive_drugs_in_exact_cmap_crosswalk"] > 0
            for row in disease_rows),
        "known_positive_pairs_in_exact_crosswalk": int(len(known_pairs)),
        "known_positive_distinct_pairs_in_exact_crosswalk": int(
            known_pairs[["diseaseId", "drugId"]].drop_duplicates().shape[0]),
        "explicit_negative_pair_count": 0,
        "disease_rows": disease_rows, "drug_rows": drug_rows,
        "trace": str(trace.path),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("audit_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               disease_route_counts=dict(disease_counts), drug_route_counts=dict(drug_counts),
               positive_pairs=result["known_positive_distinct_pairs_in_exact_crosswalk"])
    print(json.dumps({key: result[key] for key in (
        "disease_route_counts", "drug_route_counts",
        "disease_signatures_with_mapped_positive",
        "known_positive_distinct_pairs_in_exact_crosswalk", "explicit_negative_pair_count")},
        ensure_ascii=False, indent=2))
    print(f"Trace: {trace.path}")


if __name__ == "__main__":
    traced_run("external_exact_crosswalk_audit", _main, Path("artifacts/reports/traces"))
