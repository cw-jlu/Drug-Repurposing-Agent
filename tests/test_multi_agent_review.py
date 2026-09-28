"""Deterministic parts of the multi-agent LUAD literature triage (mocked LLM + PubMed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drug_repurposing_agent.llm_calls import CallStats, TracedToolCaller
from drug_repurposing_agent.multi_agent_review import (MultiAgentReviewer, apply_tier_guards,
                                                       parse_pubmed_xml, search_term,
                                                       validate_quoted_items)
from drug_repurposing_agent.trace import TraceRecorder

ABSTRACT = ("Beclomethasone dipropionate induced CYP3A5 mRNA in A549 cells.   The effect was "
            "blocked by a glucocorticoid receptor antagonist.")
RECORDS = {"111": {"pmid": "111", "title": "T", "abstract": ABSTRACT, "year": "2003",
                   "publication_types": ["Journal Article"]}}


def test_validator_keeps_verbatim_and_rejects_hallucinated_or_paraphrased_quotes():
    items = [
        {"pmid": "111", "quote": "induced CYP3A5 mRNA in A549 cells. The effect was blocked"},
        {"pmid": "999", "quote": "induced CYP3A5 mRNA in A549 cells."},
        {"pmid": "111", "quote": "BDP strongly killed A549 tumour cells in mice."},
        {"pmid": "111", "quote": "A549 cells."},
    ]
    kept, rejected = validate_quoted_items(items, RECORDS, "support")
    assert len(kept) == 1 and kept[0]["pmid"] == "111"
    assert [r["reason"] for r in rejected] == ["pmid_not_in_retrieved_set",
                                               "quote_not_verbatim_in_abstract",
                                               "quote_too_short"]


def test_parse_pubmed_xml_joins_labelled_sections():
    xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article>
      <Journal><Title>J</Title><JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue></Journal>
      <ArticleTitle>A <i>title</i></ArticleTitle><Abstract>
      <AbstractText Label="BACKGROUND">First part.</AbstractText>
      <AbstractText Label="RESULTS">Second <sub>2</sub> part.</AbstractText></Abstract>
      <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
      </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    record = parse_pubmed_xml(xml)["42"]
    assert record["title"] == "A title" and record["year"] == "2020"
    assert record["abstract"] == "BACKGROUND: First part. RESULTS: Second 2 part."


def test_tier_guards_only_downgrade():
    assert apply_tier_guards("SUPPORTED", [], [])[0] == "INSUFFICIENT_EVIDENCE"
    one = [{"claim_id": "S1", "pmid": "1", "scope": "candidate"}]
    assert apply_tier_guards("SUPPORTED", one, [])[0] == "PROMISING_BUT_INCOMPLETE"
    two = one + [{"claim_id": "S2", "pmid": "2", "scope": "candidate"}]
    assert apply_tier_guards("SUPPORTED", two, [])[0] == "SUPPORTED"
    refuted = [{"claim_id": "S2", "verdict": "refuted"}]
    assert apply_tier_guards("SUPPORTED", two, refuted)[0] == "PROMISING_BUT_INCOMPLETE"
    assert apply_tier_guards("INSUFFICIENT_EVIDENCE", two, [])[0] == "INSUFFICIENT_EVIDENCE"
    assert search_term("beclomethasone-dipropionate") == "beclomethasone dipropionate"


class FakePubMed:
    def __init__(self, records):
        self.records = records

    def search_ids(self, term, retmax=8):
        return {"query": term, "total_hits": len(self.records), "ids": list(self.records)}

    def fetch_abstracts(self, pmids):
        return {p: self.records[p] for p in pmids}


def scripted_transport(responses):
    calls = []

    def transport(payload):
        name = payload["tools"][0]["function"]["name"]
        calls.append(name)
        return {"model": "deepseek-flash", "usage": {"prompt_tokens": 5, "completion_tokens": 1},
                "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
                    {"function": {"name": name, "arguments": json.dumps(responses[name])}}]}}]}
    return transport, calls


CANDIDATE = {"rank": 2, "name": "beclomethasone-dipropionate", "identity": "exact_Hub_InChIKey",
             "pathways": ["glucocorticoid_receptor_transcription"]}


def test_full_review_rejects_hallucinated_citation_and_guards_tier(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DRUG_AGENT_TRACE_DIR", str(tmp_path / "provider"))
    responses = {
        "submit_supporting_claims": {"claims": [
            {"claim": "GR-dependent CYP3A5 induction", "pmid": "111",
             "quote": "Beclomethasone dipropionate induced CYP3A5 mRNA in A549 cells.",
             "evidence_type": "in_vitro", "scope": "candidate"},
            {"claim": "Invented efficacy", "pmid": "31415926",
             "quote": "Beclomethasone cured lung adenocarcinoma in a phase III trial.",
             "evidence_type": "clinical_trial", "scope": "candidate"}]},
        "submit_critique": {"challenges": [
            {"claim_id": "S1", "verdict": "weakened", "issue": "mechanism_not_efficacy",
             "rationale": "Target engagement only."},
            {"claim_id": "S9", "verdict": "refuted", "issue": "other", "rationale": "ghost"}],
            "contradicting_evidence": [
                {"claim": "made up", "pmid": "111", "quote": "BDP promoted tumour growth in vivo.",
                 "category": "resistance_or_harm", "scope": "candidate"}]},
        "submit_tier": {"tier": "SUPPORTED", "rationale": "Model overreach.",
                        "key_pmids": ["111", "31415926"]},
    }
    transport, calls = scripted_transport(responses)
    stats = CallStats()
    caller = TracedToolCaller(None, transport=transport, backoff_seconds=0, stats=stats)
    trace = TraceRecorder("mar_test", tmp_path / "traces")
    reviewer = MultiAgentReviewer(caller, FakePubMed(RECORDS), trace)
    result = reviewer.review(CANDIDATE)
    assert calls == ["submit_supporting_claims", "submit_critique", "submit_tier"]
    assert [c["pmid"] for c in result["validated_supporting_claims"]] == ["111"]
    assert result["rejected_claims"]["support"] == 1
    assert result["rejected_claims"]["contradiction"] == 1
    assert [c["claim_id"] for c in result["critic_challenges"]] == ["S1"]
    assert result["model_tier"] == "SUPPORTED"
    assert result["tier"] == "PROMISING_BUT_INCOMPLETE" and result["guard_notes"]
    assert result["key_pmids"] == ["111"]
    assert stats.as_dict()["attempts"] == 3
    trace_text = trace.path.read_text(encoding="utf-8")
    assert "tier_assigned" in trace_text and "support_validated" in trace_text


def test_no_records_short_circuits_without_model_calls(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DRUG_AGENT_TRACE_DIR", str(tmp_path / "provider"))
    transport, calls = scripted_transport({})
    caller = TracedToolCaller(None, transport=transport, backoff_seconds=0)
    reviewer = MultiAgentReviewer(caller, FakePubMed({}), TraceRecorder("mar_test", tmp_path))
    result = reviewer.review({"rank": 6, "name": "BRD-K84203638", "identity": "x", "pathways": []})
    assert calls == []
    assert result["tier"] == "INSUFFICIENT_EVIDENCE"
    assert result["tier_source"] == "deterministic_no_evidence"


def test_invalid_tool_output_is_retried_then_accepted(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DRUG_AGENT_TRACE_DIR", str(tmp_path / "provider"))
    state = {"n": 0}

    def transport(payload):
        state["n"] += 1
        tier = "MAYBE" if state["n"] == 1 else "INSUFFICIENT_EVIDENCE"
        return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
            {"function": {"name": "submit_tier", "arguments": json.dumps(
                {"tier": tier, "rationale": "r", "key_pmids": []})}}]}}]}

    stats = CallStats()
    caller = TracedToolCaller(None, transport=transport, backoff_seconds=0, stats=stats)
    reviewer = MultiAgentReviewer(caller, FakePubMed({}), TraceRecorder("mar_test", tmp_path))
    decision = reviewer.coordinator_agent(CANDIDATE, [], [], [])
    assert decision["tier"] == "INSUFFICIENT_EVIDENCE"
    assert stats.as_dict() == {**stats.as_dict(), "attempts": 2, "successes": 1, "failures": 1}
