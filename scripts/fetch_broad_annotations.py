"""Fetch the dated Broad Drug Repurposing Hub identity annotations."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import json
import os


BASE = "https://repo-hub.broadinstitute.org/public/data/"
FILES = ("repo-drug-annotation-20250818.txt", "repo-sample-annotation-20250818.txt")
EXPECTED_SHA256 = {
    "repo-drug-annotation-20250818.txt": "5f8284538e73c19a316d1cf45ea10300de6e15c555d75ced7c1b7387457fc523",
    "repo-sample-annotation-20250818.txt": "06eb19ad48f0e3301dbead3e184c14fd37b31c220151d5cd87fa487483db73fd",
}
DEST = Path("data/raw/repurposing_hub")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in FILES:
        path = DEST / name
        if not path.exists():
            temp = path.with_suffix(".part")
            try:
                with urlopen(BASE + name, timeout=120) as source, temp.open("wb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(block)
                os.replace(temp, path)
            except Exception:
                temp.unlink(missing_ok=True)
                raise
        actual_sha256 = sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != EXPECTED_SHA256[name]:
            raise ValueError(f"Checksum mismatch: {path}")
        manifest[name] = {"url": BASE + name, "bytes": path.stat().st_size,
                          "sha256": actual_sha256}
    Path("data/manifests/repurposing-hub-2025-08-18.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
