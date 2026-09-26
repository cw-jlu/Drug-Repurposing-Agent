"""Preserve prescore choices and redacted provider-visible traces in Git-ready files."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run, verify_trace_chain


SOURCE = Path("artifacts/reports/agent_v4_claim_choices.json")
DEST = Path("benchmark/results")
TRACE_DEST = DEST / "traces"


def copy_new(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise ValueError(f"Archive target exists: {destination}")
    if not source.is_file():
        raise FileNotFoundError(source)
    expected = sha256_file(source)
    shutil.copy2(source, destination)
    if sha256_file(destination) != expected:
        raise RuntimeError(f"Archive copy differs: {destination}")
    return {"source": str(source), "archive": str(destination), "sha256": expected}


def _main(trace: TraceRecorder) -> None:
    choices = json.loads(SOURCE.read_text(encoding="utf-8"))
    if choices.get("status") != "prescore_choices_frozen" or len(choices["choices"]) != 20:
        raise ValueError("Expected complete prescore v4 choices")
    if Path("artifacts/reports/agent_v4_claim_grade.json").exists():
        raise ValueError("Choices must be archived before scoring")
    TRACE_DEST.mkdir(parents=True, exist_ok=True)
    files = [SOURCE, Path(choices["checkpoint_file"]), Path(choices["trace_file"])]
    providers = [Path(row["provider_trace_file"]) for row in choices["choices"]]
    if len(set(providers)) != 20:
        raise ValueError("Provider traces reused")
    for row, path in zip(choices["choices"], providers):
        if sha256_file(path) != row["provider_trace_sha256"]:
            raise ValueError("Provider trace hash differs from frozen choice")
        verify_trace_chain(path, require_chain=True)
    files.extend(providers)
    trace.emit("archive_started", choices_sha256=sha256_file(SOURCE), file_count=len(files))
    copied = []
    for path in files:
        destination = DEST / SOURCE.name if path == SOURCE else (
            DEST / "agent_v4_claim_choices_checkpoint.jsonl" if path == Path(choices["checkpoint_file"])
            else TRACE_DEST / path.name)
        copied.append(copy_new(path, destination))
        trace.emit("file_archived", **copied[-1])
    receipt = {"status": "prescore_choices_and_traces_archived_before_grading",
               "choices_sha256": sha256_file(SOURCE), "files": copied,
               "archive_trace_file": str(trace.path)}
    output = DEST / "agent_v4_claim_archive_receipt.json"
    if output.exists():
        raise ValueError("Archive receipt exists")
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("archive_saved", receipt=str(output), receipt_sha256=sha256_file(output))
    print(f"Archived {len(copied)} prescore files; choices SHA-256 {receipt['choices_sha256']}")


if __name__ == "__main__":
    traced_run("agent_v4_choice_archive", _main, directory=TRACE_DEST)
