"""Fetch and verify the public ExperimentHub EH3226 Level 5 subset."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import os


URL = "https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/signatureSearchData/v0.1/lincs.h5"
DEST = Path("data/raw/GSE92742/lincs_EH3226.h5")
EXPECTED_BYTES = 2464124175
EXPECTED_SHA256 = "8087bd029d29df17ce44eb653610ad0e2b847e3f361d5540e5f798da42a473ec"


def digest(path: Path) -> str:
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists() and DEST.stat().st_size == EXPECTED_BYTES and digest(DEST) == EXPECTED_SHA256:
        print(f"Verified existing {DEST}")
        return
    temp = DEST.with_suffix(".h5.part")
    try:
        with urlopen(URL, timeout=120) as source, temp.open("wb") as target:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                target.write(block)
        if temp.stat().st_size != EXPECTED_BYTES or digest(temp) != EXPECTED_SHA256:
            raise ValueError("EH3226 file checksum/size mismatch")
        os.replace(temp, DEST)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f"Verified {DEST}")


if __name__ == "__main__":
    main()
