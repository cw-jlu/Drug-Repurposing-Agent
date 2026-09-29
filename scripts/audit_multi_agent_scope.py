"""Retrospectively audit v1 literature claims for source and exact-name scope.

This is a high-precision prefilter, not claim/quote entailment or efficacy grading.
It does not alter the frozen v1 result or call an LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.multi_agent_review import (AbstractPubMedClient,
                                                       validate_quoted_items)
from drug_repurposing_agent.trace import TraceRecorder, traced_run


SOURCE = Path("benchmark/results/multi_agent_review_v1.json")
OUTPUT_DIR = Path("artifacts/evidence_scope_audit")


def audit(trace: TraceRecorder) -> Path:
    frozen = json.loads(SOURCE.read_text(encoding="utf-8"))
    candidates = frozen["candidates"]
    pmids = sorted({str(claim["pmid"])
                    for row in candidates
                    for field in ("validated_supporting_claims", "validated_contradicting_evidence")
                    for claim in row[field]})
    trace.emit("frozen_input_loaded", source=str(SOURCE), source_sha256=sha256_file(SOURCE),
               candidate_count=len(candidates), unique_pmids=len(pmids))

    pubmed = AbstractPubMedClient()
    records: dict[str, dict] = {}
    for start in range(0, len(pmids), 20):
        batch = pmids[start:start + 20]
        fetched = pubmed.fetch_abstracts(batch)
        records.update(fetched)
        trace.emit("pubmed_batch_fetched", requested=batch, returned=sorted(fetched))

    findings = []
    for row in candidates:
        for field, kind in (("validated_supporting_claims", "support"),
                            ("validated_contradicting_evidence", "contradiction")):
            for claim in row[field]:
                kept, rejected = validate_quoted_items([claim], records, kind, row["name"])
                findings.append({"rank": row["rank"], "candidate": row["name"],
                                 "kind": kind, "pmid": str(claim["pmid"]),
                                 "scope": claim.get("scope"),
                                 "status": "passes_prefilter" if kept else "needs_review",
                                 "reason": rejected[0]["reason"] if rejected else None})

    reasons: dict[str, int] = {}
    for item in findings:
        if item["reason"]:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    result = {
        "status": "retrospective_scope_prefilter_not_semantic_entailment",
        "source": str(SOURCE), "source_sha256": sha256_file(SOURCE),
        "trace": str(trace.path), "unique_pmids": len(pmids),
        "records_returned": len(records), "claims_checked": len(findings),
        "needs_review": sum(item["status"] == "needs_review" for item in findings),
        "reasons": reasons, "findings": findings,
        "limit": ("An exact candidate name may be absent when a valid synonym is used. "
                  "Passing this prefilter does not establish that a quote supports a claim."),
    }
    output = OUTPUT_DIR / f"scope_audit_{trace.run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trace.emit("audit_saved", output=str(output), output_sha256=sha256_file(output),
               claims_checked=len(findings), needs_review=result["needs_review"], reasons=reasons)
    print(json.dumps({key: result[key] for key in
                      ("claims_checked", "needs_review", "reasons")}, ensure_ascii=False))
    print(f"Audit: {output}")
    return output


if __name__ == "__main__":
    traced_run("multi_agent_scope_audit", audit, OUTPUT_DIR / "traces")
