"""Check whether Open Targets disease cross-references safely bridge repoDB CUIs."""

from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


DISEASE = Path("artifacts/external/opentargets_26_06/disease.parquet")
EXACT = Path("benchmark/results/external_exact_crosswalk_audit.json")
REPODB = Path("benchmark/results/external_repodb_exact_crosswalk_audit.json")
OUTPUT = Path("benchmark/results/external_ot_repodb_bridge_audit.json")
CUI = re.compile(r"(?:^|:)C\d{7}$", re.IGNORECASE)


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    diseases = pd.read_parquet(DISEASE, columns=["id", "name", "dbXRefs"])
    exact = json.loads(EXACT.read_text(encoding="utf-8"))
    repodb = json.loads(REPODB.read_text(encoding="utf-8"))
    dbxref_index = {}
    prefix_counts = {}
    for identifier, xrefs in zip(diseases["id"], diseases["dbXRefs"]):
        if not pd.api.types.is_list_like(xrefs):
            continue
        references = [str(value) for value in xrefs if value is not None]
        dbxref_index[identifier] = references
        for reference in references:
            prefix = reference.split(":", 1)[0]
            prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    repodb_by_signature = {row["signature"]: row for row in repodb["rows"]}
    routes = []
    for row in exact["disease_rows"]:
        if not row["disease_id"]:
            continue
        references = dbxref_index.get(row["disease_id"], [])
        cuis = sorted(set(reference.split(":")[-1].upper() for reference in references
                          if CUI.search(reference)))
        repo_row = repodb_by_signature[row["signature"]]
        routes.append({"signature": row["signature"], "ot_id": row["disease_id"],
                       "ot_cuis": cuis, "repodb_exact_cui": repo_row["indication_id"],
                       "repo_cui_in_ot_xrefs": repo_row["indication_id"] in cuis
                       if repo_row["indication_id"] else False})
    report = {"status": "cross_reference_probe_only_no_label_merge_or_predictions",
              "source_sha256": {str(path): sha256_file(path) for path in
                                (DISEASE, EXACT, REPODB)},
              "xref_prefix_counts": prefix_counts,
              "ot_mapped_disease_signatures": len(routes),
              "with_any_umls_cui_xref": sum(bool(row["ot_cuis"]) for row in routes),
              "exact_repo_cui_confirmed_by_ot_xref": sum(
                  row["repo_cui_in_ot_xrefs"] for row in routes),
              "routes": routes, "trace": str(trace.path)}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("bridge_audit_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               with_cui=report["with_any_umls_cui_xref"])
    print(json.dumps({key: report[key] for key in (
        "ot_mapped_disease_signatures", "with_any_umls_cui_xref",
        "exact_repo_cui_confirmed_by_ot_xref", "xref_prefix_counts")}, indent=2))
    print(f"Trace: {trace.path}")


if __name__ == "__main__":
    traced_run("ot_repodb_bridge_audit", _main, Path("artifacts/reports/traces"))
