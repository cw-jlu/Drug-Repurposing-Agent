"""Grade observable LUAD research-triage claims, not biological efficacy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


def grade(report: dict, evidence: dict, signatures: list[dict]) -> dict:
    candidates = {row["rank"]: row for row in evidence["candidates"]}
    crosswalk = {int(row["rank"]): row for row in signatures}
    failures: list[str] = []
    if report.get("status") != "research_triage_only_efficacy_unverified":
        failures.append("report must abstain from efficacy claims")
    if report.get("candidate_count") != len(candidates):
        failures.append("candidate count disagrees with frozen evidence")
    seen: set[int] = set()
    allowed_actions = {"identity_resolution", "literature_review", "assay_design"}
    for item in report.get("shortlist", []):
        rank = item.get("rank")
        if not isinstance(rank, int) or rank not in candidates or rank in seen:
            failures.append(f"invalid or duplicated candidate rank: {rank}")
            continue
        seen.add(rank)
        source = candidates[rank]
        signature = crosswalk.get(rank)
        if not signature or signature["drug_name"] != source["name"] or not signature["sig_id"] or not signature["pert_id"]:
            failures.append(f"rank {rank}: source signature/name not reconciled")
        action = item.get("action")
        if action not in allowed_actions:
            failures.append(f"rank {rank}: action is not a research-only action")
        if ("mismatch" in source["identity"] or "ambiguous" in source["identity"]
                or "no_Hub_sample" in source["identity"] or "differs" in source["identity"]):
            if action == "assay_design":
                failures.append(f"rank {rank}: unresolved identity cannot proceed directly to assay")
        valid_pmids = set(source["candidate_pmids"]) | set(evidence["shared_supporting_pmids"]) | set(evidence["shared_contradicting_pmids"])
        for citation in item.get("citations", []):
            pmid, scope = citation.get("pmid"), citation.get("scope")
            if pmid not in valid_pmids:
                failures.append(f"rank {rank}: PMID {pmid} is absent from frozen evidence")
            if scope == "candidate" and pmid not in source["candidate_pmids"]:
                failures.append(f"rank {rank}: class evidence is mislabeled candidate-specific")
            if scope == "class" and pmid in source["candidate_pmids"]:
                failures.append(f"rank {rank}: candidate evidence is mislabeled class-level")
            if scope not in {"candidate", "class"}:
                failures.append(f"rank {rank}: citation scope invalid")
        if action == "literature_review" and "NR3C1" in source["targets"]:
            cited = {citation.get("pmid") for citation in item.get("citations", [])}
            if not cited.intersection(evidence["shared_contradicting_pmids"]):
                failures.append(f"rank {rank}: GR literature review omits known contradictory evidence")
    if not seen:
        failures.append("shortlist is empty")
    return {"passed": not failures, "failures": failures, "shortlist_size": len(seen),
            "checks": ["rank-to-source crosswalk integrity", "candidate/class citation scope",
                       "known GR contradiction", "identity-based assay gate",
                       "structured research-only abstention"],
            "scope": "observable one-case retrospective triage audit; not an independent holdout or treatment-efficacy score"}


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("benchmark/results/luad_llm_adjudication_v1.json"))
    parser.add_argument("--evidence", type=Path, default=Path("configs/luad_top10_evidence_v1.json"))
    parser.add_argument("--signatures", type=Path, default=Path("configs/luad_top10_signature_ids.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/luad_research_triage_grade.json"))
    args = parser.parse_args()
    trace.emit("inputs_loaded", hashes={str(path): sha256_file(path)
                                      for path in (args.report, args.evidence, args.signatures)})
    report = json.loads(args.report.read_text(encoding="utf-8"))
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    with args.signatures.open(encoding="utf-8-sig", newline="") as handle:
        signatures = list(csv.DictReader(handle))
    result = grade(report, evidence, signatures)
    result["input_sha256"] = {str(path): sha256_file(path)
                              for path in (args.report, args.evidence, args.signatures)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("grade_saved", output=str(args.output), output_sha256=sha256_file(args.output),
               passed=result["passed"], failures=result["failures"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    traced_run("luad_research_triage_grade", _main)
