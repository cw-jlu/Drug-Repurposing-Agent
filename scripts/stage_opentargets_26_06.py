"""Stage three pinned Open Targets 26.06 parquet inputs with a sealed receipt."""

from __future__ import annotations

import json
from pathlib import Path

import requests

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.06/output"
DEST = Path("artifacts/external/opentargets_26_06")
RECEIPT = Path("benchmark/results/external_opentargets_inputs_audit.json")
FILES = {
    "clinical_indication": ("clinical_indication/clinical_indication.parquet", 4970713),
    "disease": ("disease/disease.parquet", 7150787),
    "drug_molecule": ("drug_molecule/part-00000-5581d2c3-048b-4097-934f-51cec06d7ff0-c000.snappy.parquet",
                      14647216),
}


def _main(trace: TraceRecorder) -> None:
    if RECEIPT.exists():
        raise FileExistsError(f"Refusing to overwrite receipt: {RECEIPT}")
    DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (relative, expected_bytes) in FILES.items():
        url = f"{BASE}/{relative}"
        target = DEST / f"{name}.parquet"
        trace.emit("input_started", dataset=name, url=url, expected_bytes=expected_bytes,
                   output=str(target))
        if target.exists():
            if target.stat().st_size != expected_bytes:
                raise ValueError(f"Existing staged file has wrong size: {target}")
        else:
            partial = target.with_suffix(".parquet.part")
            if partial.exists():
                raise FileExistsError(f"Partial file requires inspection before retry: {partial}")
            with requests.get(url, timeout=60, stream=True) as response:
                response.raise_for_status()
                with partial.open("xb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size != expected_bytes:
                raise ValueError(f"Downloaded file differs in size: {partial}")
            partial.replace(target)
        row = {"dataset": name, "url": url, "path": str(target),
               "bytes": target.stat().st_size, "sha256": sha256_file(target)}
        rows.append(row)
        trace.emit("input_staged", **row)
        print(f"{name}: {row['bytes']} bytes; sha256={row['sha256']}")
    result = {"release": "Open Targets Platform 26.06", "status": "inputs_staged_not_scored",
              "license_check_required_before_redistribution": True,
              "datasets": rows, "trace": str(trace.path),
              "next_gate": "schema, disease/drug crosswalk, label policy and fold eligibility"}
    RECEIPT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("receipt_saved", output=str(RECEIPT), output_sha256=sha256_file(RECEIPT))
    print(f"Receipt: {RECEIPT}; trace: {trace.path}")


if __name__ == "__main__":
    traced_run("opentargets_26_06_stage", _main, Path("artifacts/reports/traces"))
