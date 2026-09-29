"""Verify the retrospective 23-case abstract review against sealed source inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run, verify_trace_chain
from scripts.build_evidence_scope_review_pack import _review_cases, SOURCE


LEDGER = Path("benchmark/results/evidence_scope_adjudication_v1.json")
OUTPUT = Path("benchmark/results/evidence_scope_adjudication_audit.json")
TRACE_DIR = Path("artifacts/evidence_scope_audit/traces")
ALLOWED = {"exclude_candidate_specific", "exclude_nonprimary", "exclude_claim_mismatch",
           "exclude_wrong_direction", "exclude_wrong_drug", "retain_narrowed",
           "retain_class_caution_only"}


def verify(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-pack", required=True, type=Path)
    args = parser.parse_args()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    pack = json.loads(args.review_pack.read_text(encoding="utf-8"))
    frozen = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len({ledger["frozen_source_sha256"], sha256_file(SOURCE),
            pack["frozen_source_sha256"]}) != 1:
        raise ValueError("Frozen source hash mismatch")
    scope_path = Path(pack["scope_audit"])
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    if sha256_file(scope_path) != pack["scope_audit_sha256"]:
        raise ValueError("Scope audit hash mismatch")
    verify_trace_chain(Path(pack["trace"]), require_chain=True)
    verify_trace_chain(Path(scope["trace"]), require_chain=True)
    expected = _review_cases(frozen, scope)
    if pack["cases"] != expected:
        raise ValueError("Review pack cases differ from sealed scope audit")
    ids = [entry["case_id"] for entry in ledger["decisions"]]
    if len(ids) != len(set(ids)) or ids != [case["case_id"] for case in expected]:
        raise ValueError("Adjudication IDs must match all 23 flagged cases in order")
    for entry in ledger["decisions"]:
        if entry["decision"] not in ALLOWED or len(entry["reason"].strip()) < 30:
            raise ValueError(f"Invalid or unexplained decision: {entry['case_id']}")
    counts = dict(sorted(Counter(entry["decision"] for entry in ledger["decisions"]).items()))
    trace.emit("inputs_verified", frozen_sha256=sha256_file(SOURCE),
               scope_sha256=sha256_file(scope_path), pack_sha256=sha256_file(args.review_pack),
               ledger_sha256=sha256_file(LEDGER), reviewed_count=len(ids), decisions=counts)
    result = {"status": ledger["status"], "frozen_source_sha256": sha256_file(SOURCE),
              "scope_audit_sha256": sha256_file(scope_path),
              "local_review_pack_sha256": sha256_file(args.review_pack),
              "adjudication_sha256": sha256_file(LEDGER), "reviewed_count": len(ids),
              "decision_counts": counts, "trace": str(trace.path),
              "limitation": "Abstract-level retrospective review; not independent expert review or clinical validation."}
    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text(encoding="utf-8"))
        comparable = {k: v for k, v in result.items() if k != "trace"}
        if {k: v for k, v in previous.items() if k != "trace"} != comparable:
            raise ValueError("Existing audit differs; refusing to overwrite")
        trace.emit("existing_output_verified", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT))
    else:
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        trace.emit("audit_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT))
    print(json.dumps(counts, ensure_ascii=False))
    print(f"Trace: {trace.path}")


if __name__ == "__main__":
    traced_run("evidence_scope_adjudication", verify, TRACE_DIR)
