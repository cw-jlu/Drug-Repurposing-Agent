"""Trace the pinned Open Targets bulk-data availability needed for v3 labels."""

from __future__ import annotations

import json
from pathlib import Path
import re

import requests

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.06/output/etl/parquet"
OUTPUT = Path("artifacts/reports/opentargets_26_06_probe.json")


def _main(trace: TraceRecorder) -> None:
    findings = []
    urls = ["https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/",
            "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.06/",
            "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.06/output/",
            "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.06/output/etl/",
            BASE + "/"]
    urls.extend("https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.06/output/" + suffix
                for suffix in ("clinical_indication/", "disease/", "drug_molecule/"))
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            names = sorted(set(re.findall(r'href="([^"]+)"', response.text,
                                          flags=re.IGNORECASE))) if response.ok else []
            row = {"url": url, "status": response.status_code,
                   "content_type": response.headers.get("Content-Type"),
                   "page_bytes": len(response.content), "links": names[:30]}
            if response.ok and any(name.endswith(".parquet") for name in names):
                files = []
                for name in names:
                    if not name.endswith(".parquet"):
                        continue
                    head = requests.head(url + name, timeout=30)
                    files.append({"name": name, "status": head.status_code,
                                  "bytes": head.headers.get("Content-Length")})
                row["files"] = files
        except requests.RequestException as exc:
            row = {"url": url, "error_type": type(exc).__name__}
        findings.append(row)
        trace.emit("release_path_probed", **row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = {"release": "26.06", "status": "availability_probe_not_labels_staged",
              "paths": findings, "trace": str(trace.path)}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("probe_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT))
    print(json.dumps(findings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    traced_run("opentargets_26_06_probe", _main, Path("artifacts/reports/traces"))
