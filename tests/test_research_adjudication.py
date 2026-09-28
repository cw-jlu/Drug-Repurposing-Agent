"""Research-only LLM output must remain inside the frozen evidence boundary."""

from __future__ import annotations

import json

import pytest

from drug_repurposing_agent.research_adjudication import adjudicate


def evidence() -> dict:
    return {
        "candidates": [
            {"rank": rank,
             "identity": "exact_Hub_InChIKey" if rank == 2 else "Hub_structure_mismatch",
             "candidate_pmids": ["12538830"] if rank == 2 else [],
             "confidence": "insufficient_evidence"}
            for rank in range(1, 11)
        ],
        "shared_supporting_pmids": ["12204894"],
        "shared_contradicting_pmids": ["24736075"],
    }


def answer(shortlist: list[dict], finish_reason: str = "stop"):
    return {"model": "mock", "choices": [{"finish_reason": finish_reason,
            "message": {"tool_calls": [{"function": {
                "name": "submit_research_triage",
                "arguments": json.dumps({"shortlist": shortlist})}}]}}],
            "usage": {"total_tokens": 100}}


def row(rank: int = 2, action: str = "preclinical_assay",
        citations: list[dict] | None = None) -> dict:
    return {"rank": rank, "action": action,
            "citations": citations if citations is not None else
            [{"pmid": "12538830", "scope": "candidate"}],
            "reason": "Research-only follow-up; efficacy unverified."}


def test_accepts_bounded_research_action_without_test_labels():
    seen = {}
    def transport(payload):
        seen.update(payload)
        return answer([row()])
    result = adjudicate(evidence(), "Reviewed evidence matrix", transport)
    assert result["status"] == "research_triage_only_efficacy_unverified"
    assert result["shortlist"][0]["rank"] == 2
    assert seen["tools"][0]["function"]["strict"] is True
    assert "ratings" not in json.dumps(seen)


@pytest.mark.parametrize("bad_row", [
    row(rank=11),
    row(rank=1),
    row(citations=[{"pmid": "99999999", "scope": "candidate"}]),
    row(citations=[{"pmid": "12204894", "scope": "candidate"}]),
    row(action="clinical_treatment"),
])
def test_rejects_unknown_or_unsafe_model_decisions(bad_row):
    with pytest.raises(ValueError):
        adjudicate(evidence(), "Reviewed evidence matrix",
                   lambda _: answer([bad_row]))


def test_rejects_truncated_response():
    with pytest.raises(ValueError, match="Truncated"):
        adjudicate(evidence(), "Reviewed evidence matrix",
                   lambda _: answer([row()], finish_reason="length"))
