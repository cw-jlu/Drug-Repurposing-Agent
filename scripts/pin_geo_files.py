"""Pin GEO files for a registry entry without keeping them: stream, hash, discard.

    python -m scripts.pin_geo_files URL [URL ...]

Prints {url: {"bytes": n, "sha256": hex}} so an entry can be registered before any
data is stored locally; the agent's fetch_geo_series downloads the file only when a
request needs it and refuses it unless the bytes match this pin.
"""

from __future__ import annotations

from hashlib import sha256
import json
import sys
import urllib.request

from drug_repurposing_agent.geo_cohort import USER_AGENT
from drug_repurposing_agent.trace import TraceRecorder, traced_run


def pin(url: str) -> dict:
    h, n = sha256(), 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        while chunk := response.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
    return {"bytes": n, "sha256": h.hexdigest()}


def _main(trace: TraceRecorder) -> None:
    result = {}
    for url in sys.argv[1:]:
        result[url] = pin(url)
        trace.emit("pinned", url=url, **result[url])
    print(json.dumps(result, indent=1))


def main() -> None:
    traced_run("pin_geo_files", _main)


if __name__ == "__main__":
    main()
