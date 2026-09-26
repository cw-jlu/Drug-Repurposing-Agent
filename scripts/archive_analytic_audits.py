"""Archive completed B1k all-seed and external-input audits with their traces."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run, verify_trace_chain


REPORT_DIR = Path("artifacts/reports")
TRACE_DIR = Path("artifacts/traces")
DEST = Path("benchmark/results")


def copy_new(source: Path, destination: Path) -> dict:
    if not source.is_file() or destination.exists():
        raise ValueError(f"Missing source or existing archive target: {source} -> {destination}")
    digest = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(destination) != digest:
        raise RuntimeError(f"Archive copy differs: {destination}")
    return {"source": str(source), "archive": str(destination), "sha256": digest}


def _main(trace: TraceRecorder) -> None:
    reports = [REPORT_DIR / f"b1k_all_seeds_{split}.json" for split in ("random", "weak")]
    reports.extend([REPORT_DIR / "external_holdout_source_audit.json",
                    REPORT_DIR / "external_staged_inputs_audit.json"])
    files = []
    for report_path in reports[:2]:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["status"] != "retrospective_diagnostic_not_new_benchmark":
            raise ValueError(f"Unexpected B1k report status: {report_path}")
        if report["count"] != 100 or report["summary"]["seed_count"] != 100:
            raise ValueError(f"All frozen B1k seeds are required: {report_path}")
        checkpoint = Path(report["checkpoint_file"])
        if sha256_file(checkpoint) != report["checkpoint_sha256"]:
            raise ValueError(f"Seed checkpoint hash differs: {checkpoint}")
        source_trace = Path(report["trace_file"])
        verify_trace_chain(source_trace, require_chain=True)
        files.extend([report_path, checkpoint, source_trace])
    for index, trace_pattern, expected_status in (
        (2, "*_external_holdout_source_audit_*.jsonl",
         "external_holdout_not_ready_no_model_outcomes_computed"),
        (3, "*_external_staged_inputs_audit_*.jsonl",
         "disease_inputs_staged_drug_matrix_and_labels_absent"),
    ):
        external = json.loads(reports[index].read_text(encoding="utf-8"))
        if external["status"] != expected_status:
            raise ValueError(f"Unexpected external audit status: {reports[index]}")
        external_traces = sorted(TRACE_DIR.glob(trace_pattern))
        if len(external_traces) != 1:
            raise ValueError(f"Expected exactly one external audit trace: {trace_pattern}")
        verify_trace_chain(external_traces[0], require_chain=True)
        files.extend([reports[index], external_traces[0]])
    if len(set(files)) != len(files):
        raise ValueError("Audit files overlap")
    trace.emit("archive_started", source_file_count=len(files), report_hashes={
        str(path): sha256_file(path) for path in reports})
    copies = []
    for source in files:
        target = DEST / "traces" / source.name if source.parent == TRACE_DIR else DEST / source.name
        copies.append(copy_new(source, target))
        trace.emit("file_archived", **copies[-1])
    receipt_path = DEST / "analytic_audits_archive_receipt.json"
    if receipt_path.exists():
        raise ValueError("Archive receipt already exists")
    receipt = {"status": "all_seed_diagnostics_and_source_gate_archived",
               "files": copies, "archive_trace_file": str(trace.path)}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    trace.emit("receipt_saved", path=str(receipt_path), sha256=sha256_file(receipt_path))
    print(f"Archived {len(copies)} audit files")


if __name__ == "__main__":
    traced_run("analytic_audits_archive", _main, directory=DEST / "traces")
