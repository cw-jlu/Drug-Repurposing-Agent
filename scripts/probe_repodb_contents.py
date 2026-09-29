"""Trace repoDB schema/value domains before any external prediction scoring."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


SOURCE = Path("artifacts/external/repodb_2017/shiny.RData")
OUTPUT = Path("benchmark/results/external_repodb_2017_domains.json")


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    sys.path.insert(0, str(Path("artifacts/tools/python_packages").resolve()))
    import rdata
    frame = rdata.read_rda(SOURCE, default_encoding="utf-8",
                           force_default_encoding=True)["drug.fr"]
    domains = {}
    for field in ("status", "phase", "sem_type"):
        domains[field] = {str(key): int(value) for key, value in
                          frame[field].value_counts(dropna=False).items()}
    for field in ("TrialStatus", "DetailedStatus"):
        counts = frame[field].value_counts(dropna=False)
        domains[field] = {"distinct_values": int(len(counts)),
                          "top_five": {str(key): int(value) for key, value in
                                       counts.head(5).items()}}
    sample = frame[["Drug", "Indication", "drug_name", "drug_id", "ind_name",
                    "ind_id", "TrialStatus", "status"]].head(6).fillna("").to_dict("records")
    result = {"status": "domains_only_no_predictions", "source_sha256": sha256_file(SOURCE),
              "rows": len(frame), "domains": domains, "first_six_rows": sample,
              "trace": str(trace.path)}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("domains_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               rows=len(frame), domains=domains)
    print(json.dumps({"rows": len(frame), "domains": domains, "first_six_rows": sample},
                     ensure_ascii=True, indent=2))
    print(f"Trace: {trace.path}")


if __name__ == "__main__":
    traced_run("repodb_2017_domains", _main, Path("artifacts/reports/traces"))
