"""Inspect the staged public CMap signature object without scoring any method."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


sys.path.insert(0, str(Path("artifacts/tools/python_packages").resolve()))
import rdata  # noqa: E402


MATRIX = Path("artifacts/external/cdrpipe_data/cmap_signatures.RData")
META = Path("artifacts/external/cdrpipe-comparative-analysis/drug_signatures/data/cmap/cmap_drug_experiments_new.csv")
OUTPUT = Path("benchmark/results/external_cmap_matrix_alignment_audit.json")
PROBES = Path("artifacts/external/cdrpipe-comparative-analysis/drug_signatures/data/cmap/cmap_probe_set_ids.csv")


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    trace.emit("input_verified", matrix=str(MATRIX), matrix_bytes=MATRIX.stat().st_size,
               matrix_sha256=sha256_file(MATRIX), metadata_sha256=sha256_file(META),
               rdata_version=getattr(rdata, "__version__", "unknown"))
    objects = rdata.read_rda(MATRIX)
    summary = {}
    for key, value in objects.items():
        shape = list(value.shape) if hasattr(value, "shape") else None
        summary[key] = {"python_type": type(value).__name__, "shape": shape,
                        "index_sample": [str(v) for v in value.index[:5]]
                        if hasattr(value, "index") else None,
                        "column_sample": [str(v) for v in value.columns[:5]]
                        if hasattr(value, "columns") else None}
        if isinstance(value, pd.DataFrame):
            metadata = pd.read_csv(META)
            probe_ids = pd.read_csv(PROBES)["probe_set_id"]
            first_values = pd.to_numeric(value.iloc[:, 0], errors="coerce")
            summary[key].update({
                "first_three_rows_first_three_columns": value.iloc[:3, :3].astype(str).values.tolist(),
                "first_column_all_matches_probe_ids": bool(
                    len(first_values) == len(probe_ids) and
                    first_values.reset_index(drop=True).eq(probe_ids).all()),
                "metadata_rows": len(metadata), "metadata_columns": list(metadata.columns),
                "metadata_id_first_last": [str(metadata["id"].iloc[0]),
                                           str(metadata["id"].iloc[-1])],
                "probe_rows": len(probe_ids), "probe_id_first_last": [str(probe_ids.iloc[0]),
                                                                  str(probe_ids.iloc[-1])],
                "matrix_numeric_min_max": [float(value.iloc[:, 1:].min().min()),
                                           float(value.iloc[:, 1:].max().max())],
                "matrix_null_cells": int(value.iloc[:, 1:].isna().sum().sum())})
        trace.emit("matrix_object_loaded", object_name=key, **summary[key])
    report = {"status": "raw_cmap_matrix_staged_and_read_not_benchmark_scored",
              "source": "https://ucsf.box.com/s/m54ipylmdytjsqmlp7axnabvjh2q8lwl",
              "box_file_id": "f_2206971964738", "matrix_bytes": MATRIX.stat().st_size,
              "matrix_sha256": sha256_file(MATRIX), "metadata_sha256": sha256_file(META),
              "probe_ids_sha256": sha256_file(PROBES),
              "reader": f"rdata {getattr(rdata, '__version__', 'unknown')}",
              "objects": summary, "trace": str(trace.path),
              "not_yet_verified": ["gene/disease crosswalk", "drug/indication crosswalk",
                                   "positive/negative label policy", "eligible independent units"]}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("audit_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Trace: {trace.path}")


if __name__ == "__main__":
    traced_run("external_cmap_matrix_audit", _main, Path("artifacts/reports/traces"))
