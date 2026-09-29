"""Create a traced, local-only source pack for the flagged v1 literature claims.

The pack contains full PubMed abstracts for adjudication and remains Git-ignored.
It does not change the frozen v1 grades or assert claim correctness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import sleep

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.multi_agent_review import AbstractPubMedClient
from drug_repurposing_agent.trace import TraceRecorder, traced_run, verify_trace_chain


SOURCE = Path("benchmark/results/multi_agent_review_v1.json")
OUTPUT_DIR = Path("artifacts/evidence_scope_audit")


def _review_cases(frozen: dict, scope_audit: dict) -> list[dict]:
    ordered = []
    for row in frozen["candidates"]:
        for field, kind in (("validated_supporting_claims", "support"),
                            ("validated_contradicting_evidence", "contradiction")):
            for index, claim in enumerate(row[field]):
                ordered.append((row, kind, index, claim))
    findings = scope_audit["findings"]
    if len(ordered) != len(findings) or len(findings) != 65:
        raise ValueError("Frozen claim and scope-audit counts do not match")
    cases = []
    for (row, kind, index, claim), finding in zip(ordered, findings):
        if (row["rank"], row["name"], kind, str(claim["pmid"]), claim["scope"]) != (
                finding["rank"], finding["candidate"], finding["kind"],
                str(finding["pmid"]), finding["scope"]):
            raise ValueError("Scope-audit finding does not match frozen claim order")
        if finding["status"] != "needs_review":
            continue
        cases.append({"case_id": f"rank{row['rank']:02d}_{kind}_{index:02d}",
                      "rank": row["rank"], "candidate": row["name"],
                      "kind": kind, "pmid": str(claim["pmid"]),
                      "claimed_scope": claim["scope"], "claim": claim["claim"],
                      "quote": claim["quote"], "prefilter_reason": finding["reason"]})
    if len(cases) != 23:
        raise ValueError(f"Expected 23 flagged claims, found {len(cases)}")
    return cases


def _fetch_with_retry(client: AbstractPubMedClient, pmids: list[str],
                      trace: TraceRecorder) -> dict[str, dict]:
    for attempt in range(1, 4):
        try:
            records = client.fetch_abstracts(pmids)
            trace.emit("pubmed_batch_fetched", requested=pmids,
                       returned=sorted(records), attempt=attempt)
            return records
        except Exception as exc:
            trace.emit("pubmed_batch_failed", requested=pmids, attempt=attempt,
                       error_type=type(exc).__name__, error=str(exc))
            if attempt == 3:
                raise
            sleep(attempt)
    raise AssertionError("Unreachable retry state")


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope-audit", type=Path, required=True)
    args = parser.parse_args()
    scope = json.loads(args.scope_audit.read_text(encoding="utf-8"))
    frozen = json.loads(SOURCE.read_text(encoding="utf-8"))
    if scope["source_sha256"] != sha256_file(SOURCE):
        raise ValueError("Frozen v1 source hash differs from scope audit")
    source_trace = Path(scope["trace"])
    verify_trace_chain(source_trace, require_chain=True)
    cases = _review_cases(frozen, scope)
    trace.emit("inputs_validated", source_sha256=sha256_file(SOURCE),
               scope_audit_sha256=sha256_file(args.scope_audit),
               scope_trace_sha256=sha256_file(source_trace), case_count=len(cases))
    client = AbstractPubMedClient()
    records: dict[str, dict] = {}
    pmids = sorted({case["pmid"] for case in cases})
    for start in range(0, len(pmids), 20):
        records.update(_fetch_with_retry(client, pmids[start:start + 20], trace))
    missing = sorted(set(pmids) - set(records))
    if missing:
        raise ValueError(f"PubMed returned no record for PMIDs: {missing}")
    pack = {"status": "local_source_pack_for_preliminary_review_not_clinical_adjudication",
            "frozen_source": str(SOURCE), "frozen_source_sha256": sha256_file(SOURCE),
            "scope_audit": str(args.scope_audit),
            "scope_audit_sha256": sha256_file(args.scope_audit),
            "source_trace_sha256": sha256_file(source_trace),
            "cases": cases, "pubmed_records": records, "trace": str(trace.path),
            "limitation": "Current PubMed snapshot may differ from the prior audit; full abstracts stay local."}
    output = OUTPUT_DIR / f"review_pack_{trace.run_id}.json"
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}")
    output.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("review_pack_saved", output=str(output), output_sha256=sha256_file(output),
               case_count=len(cases), unique_pmids=len(pmids))
    print(f"Review pack: {output}; cases={len(cases)}; PMIDs={len(pmids)}")


if __name__ == "__main__":
    traced_run("evidence_scope_review_pack", _main, OUTPUT_DIR / "traces")
