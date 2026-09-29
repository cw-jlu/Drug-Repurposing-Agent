"""Trace public RePairDB file availability without downloading or scoring it."""

from __future__ import annotations

import json
from pathlib import Path
import re

import requests

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


DOI = "10.17632/8wjx8yb55n.1"
URL = "https://api.data.mendeley.com/datasets/publics/8wjx8yb55n/files?version=1"
OUTPUT = Path("benchmark/results/external_repairdb_source_probe.json")


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    response = requests.get(URL, timeout=30,
                            headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"})
    trace.emit("public_metadata_fetched", url=URL, http_status=response.status_code,
               bytes=len(response.content))
    files = []
    page_status = None
    if response.ok:
        data = response.json()
        entries = data if isinstance(data, list) else data.get("files", [])
        files = [{key: item.get(key) for key in ("id", "filename", "size", "content_type")}
                 for item in entries]
    else:
        page_url = "https://data.mendeley.com/datasets/8wjx8yb55n/1"
        page = requests.get(page_url, timeout=30)
        page_status = page.status_code
        trace.emit("public_page_fetched", url=page_url, http_status=page.status_code,
                   bytes=len(page.content))
        if page.ok:
            files = [{"filename_mention": name} for name in sorted(set(
                re.findall(r"[A-Za-z0-9_.-]+\.(?:csv|parquet|zip|pdf|py)", page.text,
                           flags=re.IGNORECASE)))]
    result = {"status": "metadata_only_not_staged_or_scored", "doi": DOI,
              "url": URL, "version": 1, "api_status": response.status_code,
              "page_status": page_status, "files": files,
              "trace": str(trace.path)}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("probe_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               file_count=len(files))
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    traced_run("repairdb_public_source_probe", _main, Path("artifacts/reports/traces"))
