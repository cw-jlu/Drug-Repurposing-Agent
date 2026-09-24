"""Download the small GEO inputs used by the LUAD audit; excludes Level 5 GCTX."""

from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen


MANIFESTS = [Path("data/manifests/gse32863.json"),
             Path("data/manifests/gse92742-a549.json")]


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    for manifest_path in MANIFESTS:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = Path("data/raw") / manifest["accession"]
        target.mkdir(parents=True, exist_ok=True)
        for filename, source in manifest["sources"].items():
            path = target / filename
            if not path.exists():
                print(f"Downloading {filename}", flush=True)
                with urlopen(source["url"], timeout=120) as response, path.open("wb") as out:
                    while block := response.read(1024 * 1024):
                        out.write(block)
            if path.stat().st_size != source["bytes"] or digest(path) != source["sha256"]:
                raise ValueError(f"Verification failed: {path}")
            print(f"Verified {path}")


if __name__ == "__main__":
    main()
