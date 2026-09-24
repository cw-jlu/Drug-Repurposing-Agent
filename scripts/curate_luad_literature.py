"""Attach narrowly reviewed PubMed context without upgrading efficacy tier."""

from pathlib import Path
import json

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.evidence import CandidateLedger, Citation


ROOT = Path("artifacts/reports/luad_eh3226")
CURATED = {
    "hydrocortisone": Citation(
        kind="PMID", identifier="35676421", direction="context",
        evidence_type="computational_NSCLC_hypothesis",
        note="Independent network analysis proposes hydrocortisone by name for NSCLC; no experimental or clinical efficacy validation is reported in the abstract, and the EH3226 compound identity is unresolved."),
    "beclomethasone-dipropionate": Citation(
        kind="PMID", identifier="12538830", direction="context",
        evidence_type="A549_mechanistic_in_vitro",
        note="A549 experiments report glucocorticoid receptor-mediated CYP3A5 induction by beclomethasone dipropionate; this is a transcriptional mechanism observation, not anticancer efficacy."),
}


def main() -> None:
    search_path = ROOT / "pubmed_search.json"
    search = json.loads(search_path.read_text(encoding="utf-8"))
    if search["ranking_sha256"] != sha256_file(ROOT / "all_candidates.csv"):
        raise ValueError("PubMed search was not run on the current ranking")
    hits = {q["drug_name"]: {r["pmid"] for r in q["records"]}
            for q in search["queries"]}
    for name, citation in CURATED.items():
        citation.validate()
        if citation.identifier not in hits.get(name, set()):
            raise ValueError(f"Reviewed PMID missing from frozen search: {name}")
    output = []
    for ledger_path in sorted((ROOT / "evidence_ledger").glob("rank_*.json")):
        raw = json.loads(ledger_path.read_text(encoding="utf-8"))
        name = raw["drug_name"]
        for field in ("supporting_evidence", "contradicting_evidence", "context_evidence"):
            raw[field] = [Citation(**entry) for entry in raw.get(field, [])]
        ledger = CandidateLedger(**raw)
        if name in CURATED:
            ledger.context_evidence = [CURATED[name]]
            ledger.limitations = [x for x in ledger.limitations
                                  if x != "No target, safety, or literature evidence has been curated."]
            ledger.limitations.append("No compound-verified clinical efficacy or safety evidence has been curated.")
        ledger.save(ledger_path)
        if name in CURATED:
            output.append({"drug_name": name, "rank": ledger.transcriptomic_rank,
                           "pmid": CURATED[name].identifier,
                           "evidence_type": CURATED[name].evidence_type})
    (ROOT / "literature_triage_manifest.json").write_text(json.dumps({
        "pubmed_search_sha256": sha256_file(search_path),
        "reviewed_context": output,
        "rule": "Context does not upgrade efficacy confidence; other hits remain unreviewed.",
    }, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
