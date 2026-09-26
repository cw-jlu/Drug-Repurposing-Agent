"""Inventory sparse-checked-out external disease inputs without scoring outcomes."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


ROOT = Path("artifacts/external/cdrpipe-comparative-analysis")
OUTPUT = Path("artifacts/reports/external_staged_inputs_audit.json")
EXPECTED_REVISION = "f5b358ec5f86c53a0043789bbcec0052c662c438"


def audit_disease(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not {"gene_symbol", "mean_logfc", "organism"}.issubset(columns):
            raise ValueError(f"Unexpected disease signature columns: {path}")
        rows, valid = 0, 0
        for row in reader:
            rows += 1
            try:
                usable = bool(row["gene_symbol"]) and math.isfinite(float(row["mean_logfc"]))
            except (TypeError, ValueError):
                usable = False
            valid += int(usable)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path),
            "rows": rows, "finite_gene_logfc_rows": valid}


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise ValueError("Staged-input audit exists; refusing overwrite")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != EXPECTED_REVISION:
        raise ValueError(f"External revision changed: {revision}")
    files = sorted((ROOT / "creeds/data/manual_signatures_extracted").glob("*.csv"))
    if len(files) != 233:
        raise ValueError(f"Expected 233 staged disease CSVs, found {len(files)}")
    trace.emit("staged_inputs_found", upstream_revision=revision, disease_file_count=len(files))
    diseases = [audit_disease(path) for path in files]
    metadata = []
    for relative in ("drug_signatures/data/cmap/cmap_drug_experiments_new.csv",
                     "shared/gene_id_conversion_table.tsv"):
        path = ROOT / relative
        metadata.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    if any(row["finite_gene_logfc_rows"] == 0 for row in diseases):
        raise ValueError("At least one staged disease signature has no finite expression rows")
    report = {"status": "disease_inputs_staged_drug_matrix_and_labels_absent",
              "source": "https://github.com/enockniyonkuru/cdrpipe-comparative-analysis",
              "upstream_revision": revision,
              "disease_file_count": len(diseases),
              "disease_total_rows": sum(row["rows"] for row in diseases),
              "disease_total_finite_gene_logfc_rows": sum(row["finite_gene_logfc_rows"] for row in diseases),
              "disease_signatures": diseases, "metadata": metadata,
              "required_missing_for_v3": ["raw drug-expression matrix", "known-indication labels",
                                          "frozen drug/disease ID mapping and split policy"],
              "scope": "Input provenance only; no model outcomes or provider calls",
              "trace_file": str(trace.path)}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("staged_inputs_audited", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               disease_file_count=len(diseases), total_rows=report["disease_total_rows"],
               finite_rows=report["disease_total_finite_gene_logfc_rows"], ready=False)
    print(json.dumps({key: report[key] for key in ("status", "upstream_revision",
                                                 "disease_file_count", "disease_total_rows",
                                                 "disease_total_finite_gene_logfc_rows")}, indent=2))


if __name__ == "__main__":
    traced_run("external_staged_inputs_audit", _main)
