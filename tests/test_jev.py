import io
import json

import pytest

from drug_repurposing_agent.jev import (
    JevClient, choice_question, gate_choice, noul_question, score_question, validate_answer,
)


def test_jev_question_types_and_answer_validation():
    choice = choice_question("Next step?", {"continue": "data valid", "review": "uncertain"})
    score = score_question("Data quality?", ["poor", "partial", "good"])
    noul = noul_question("Enough evidence?", "yes", "no")
    validate_answer(choice, {"type": "choice", "choice": "continue", "confidence": 0.94,
                             "probabilities": {"continue": 0.94, "review": 0.06}})
    validate_answer(score, {"type": "score", "score": 1.7, "confidence": 0.9,
                            "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}})
    validate_answer(noul, {"type": "noul", "noul": 0.82})


def test_jev_rejects_undeclared_choice_or_bad_distribution():
    question = choice_question("Next step?", {"continue": "good", "review": "uncertain"})
    with pytest.raises(ValueError):
        validate_answer(question, {"type": "choice", "choice": "publish_clinical",
                                   "confidence": 1.0,
                                   "probabilities": {"continue": 0.8, "review": 0.2}})
    with pytest.raises(ValueError):
        validate_answer(question, {"type": "choice", "choice": "continue",
                                   "confidence": 0.9,
                                   "probabilities": {"continue": 0.4, "review": 0.4}})


def test_jev_low_confidence_and_unavailable_abstain():
    question = choice_question("Next step?", {"continue": "good", "review": "uncertain"})
    answer = {"type": "choice", "choice": "continue", "confidence": 0.79,
              "probabilities": {"continue": 0.79, "review": 0.21}}
    assert gate_choice(question, answer).action == "manual_review"
    assert gate_choice(question, None).source == "rule_fallback"
    assert gate_choice(question, {**answer, "confidence": 0.85}, high_risk=True).action == "manual_review"


def test_jev_cannot_publish_clinical_claim():
    question = choice_question("Next step?", {"publish_clinical": "publish", "review": "wait"})
    answer = {"type": "choice", "choice": "publish_clinical", "confidence": 0.99,
              "probabilities": {"publish_clinical": 0.99, "review": 0.01}}
    result = gate_choice(question, answer)
    assert result.action == "manual_review"
    assert result.source == "hard_rule"


def test_jev_http_request_and_response_are_schema_checked(monkeypatch):
    seen = []
    def fake_urlopen(request, timeout):
        seen.append((request, timeout))
        payload = {"model": "jev-version", "usage": {"input_tokens": 2, "output_tokens": 1},
                   "answers": {"route": {"type": "noul", "noul": 0.9}}}
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr("drug_repurposing_agent.jev.urlopen", fake_urlopen)
    question = noul_question("Enough data?", "yes", "no")
    result = JevClient("secret-key").ask({"pairs": 57}, {"route": question})
    assert result["answers"]["route"]["noul"] == 0.9
    assert seen[0][0].get_header("Authorization") == "Bearer secret-key"
    assert json.loads(seen[0][0].data)["questions"]["route"] == question


def test_from_env_routes_opencode_key_to_zen(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-test")
    monkeypatch.chdir(tmp_path)
    client = JevClient.from_env()
    assert client.api_url.startswith("https://opencode.ai/zen/") and client.model == "jev-1.13"
    assert client.provider == "opencode_zen_jev"


def test_request_sends_user_agent(monkeypatch):
    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b"{}"

    def fake_urlopen(request, timeout):
        seen["ua"] = request.get_header("User-agent")
        raise RuntimeError("stop")

    monkeypatch.setattr("drug_repurposing_agent.jev.urlopen", fake_urlopen)
    question = choice_question("Route", {"a": "A", "b": "B"})
    try:
        JevClient("k", api_url="https://opencode.ai/zen/v1/systemone").ask({"x": 1}, {"q": question})
    except RuntimeError:
        pass
    assert seen["ua"].startswith("drug-repurposing-agent/")
