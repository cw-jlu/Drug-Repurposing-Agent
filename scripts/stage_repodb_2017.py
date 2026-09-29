"""Stage and inspect the original public repoDB app data at a pinned commit."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import requests

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


REVISION = "cd4edec67e57df1d2a5f552e30a6960a38d79432"
URL = ("https://raw.githubusercontent.com/adam-sam-brown/repoDB/"
       f"{REVISION}/Shiny_Application/data/shiny.RData")
TARGET = Path("artifacts/external/repodb_2017/shiny.RData")
OUTPUT = Path("benchmark/results/external_repodb_2017_source_audit.json")


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        response = requests.get(URL, timeout=40)
        trace.emit("source_fetched", url=URL, status=response.status_code,
                   bytes=len(response.content))
        response.raise_for_status()
        if len(response.content) != 306051:
            raise ValueError("Pinned Git blob size differs")
        TARGET.write_bytes(response.content)
    if TARGET.stat().st_size != 306051:
        raise ValueError("Staged repoDB blob size differs")
    sys.path.insert(0, str(Path("artifacts/tools/python_packages").resolve()))
    import rdata
    objects = rdata.read_rda(TARGET, default_encoding="utf-8",
                             force_default_encoding=True)
    summary = {name: {"type": type(value).__name__,
                      "shape": list(value.shape) if hasattr(value, "shape") else None,
                      "columns": list(value.columns) if hasattr(value, "columns") else None}
               for name, value in objects.items()}
    report = {"status": "source_staged_schema_only_not_label_crosswalked",
              "source": URL, "revision": REVISION, "bytes": TARGET.stat().st_size,
              "sha256": sha256_file(TARGET), "rdata_version": rdata.__version__,
              "objects": summary, "trace": str(trace.path)}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("audit_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               objects=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Trace: {trace.path}")


if __name__ == "__main__":
    traced_run("repodb_2017_source_stage", _main, Path("artifacts/reports/traces"))
