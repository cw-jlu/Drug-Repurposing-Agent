"""Fetch and verify the official TRANSCRIPT 2.0.0 archive."""

from datetime import date
from hashlib import md5, sha256
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


URL = "https://zenodo.org/api/records/7982976/files/TRANSCRIPT_dataset_v2.0.0.zip/content"
MD5 = "67b5be71611361ca493303b052a4944c"
EXPECTED = {"items.csv", "users.csv", "ratings_mat.csv", "README", "LICENSE"}


def digest(path: Path, algorithm) -> str:
    h = algorithm()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "TRANSCRIPT_dataset_v2.0.0.zip"
    if not archive.exists():
        with urlopen(URL, timeout=120) as response, archive.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
    actual = digest(archive, md5)
    if actual != MD5:
        raise ValueError(f"TRANSCRIPT MD5 mismatch: {actual}")
    with ZipFile(archive) as z:
        members = {Path(name).name for name in z.namelist() if not name.endswith("/")}
        if members != EXPECTED:
            raise ValueError(f"Unexpected archive members: {members}")
        for member in z.infolist():
            parts = Path(member.filename).parts
            if ".." in parts or Path(member.filename).is_absolute():
                raise ValueError("Unsafe archive path")
        z.extractall(raw)
    extracted = raw / "TRANSCRIPT_dataset_v2.0.0"
    manifest = {
        "dataset": "TRANSCRIPT", "version": "2.0.0", "zenodo_record": 7982976,
        "source_url": URL, "archive_name": archive.name,
        "download_checked_at": date.today().isoformat(),
        "archive_bytes": archive.stat().st_size, "archive_md5": actual,
        "archive_sha256": digest(archive, sha256),
        "files": {name: {"bytes": (extracted / name).stat().st_size,
                         "sha256": digest(extracted / name, sha256)} for name in sorted(EXPECTED)},
        "license": "See original LICENSE in verified archive",
    }
    destination = Path("data/manifests/transcript-v2.0.0.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {archive}; wrote {destination}")


if __name__ == "__main__":
    main()
