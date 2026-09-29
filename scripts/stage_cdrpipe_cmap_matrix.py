"""Stage the public CDRPipe CMap matrix with size/hash and a chained trace."""

from __future__ import annotations

from pathlib import Path
import subprocess

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


SOURCE = ("https://ucsf.box.com/index.php?rm=box_download_shared_file"
          "&shared_name=m54ipylmdytjsqmlp7axnabvjh2q8lwl&file_id=f_2206971964738")
TARGET = Path("artifacts/external/cdrpipe_data/cmap_signatures.RData")
EXPECTED_BYTES = 243722737


def _main(trace: TraceRecorder) -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    current_bytes = TARGET.stat().st_size if TARGET.exists() else 0
    if current_bytes > EXPECTED_BYTES:
        raise ValueError("Existing matrix is larger than published Box metadata; refusing overwrite")
    trace.emit("source_pinned", url=SOURCE, file_id="f_2206971964738",
               expected_bytes=EXPECTED_BYTES, prior_bytes=current_bytes,
               output=str(TARGET))
    if current_bytes != EXPECTED_BYTES:
        command = ["curl.exe", "--location", "--fail", "--silent", "--show-error",
                   "--retry", "3", "--retry-all-errors", "--continue-at", "-",
                   "--max-time", "900", "--output", str(TARGET), SOURCE]
        result = subprocess.run(command, capture_output=True, text=True, timeout=930,
                                check=False)
        after = TARGET.stat().st_size if TARGET.exists() else 0
        trace.emit("download_finished", exit_code=result.returncode, bytes=after,
                   stderr_present=bool(result.stderr))
        if result.returncode:
            raise RuntimeError(f"CMap matrix download failed with curl exit {result.returncode}; "
                               f"partial bytes retained: {after}")
    size = TARGET.stat().st_size
    if size != EXPECTED_BYTES:
        raise ValueError(f"CMap matrix size differs: {size} != {EXPECTED_BYTES}")
    trace.emit("matrix_staged", output=str(TARGET), bytes=size,
               sha256=sha256_file(TARGET))
    print(f"Staged {TARGET}; bytes={size}; sha256={sha256_file(TARGET)}; trace={trace.path}")


if __name__ == "__main__":
    traced_run("cdrpipe_cmap_matrix_stage", _main,
               Path("artifacts/reports/traces"))
