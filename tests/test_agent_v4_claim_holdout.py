from evals.agent_v4_claim_common import load_holdout, public_case
from evals.grade_agent_v4_claim_holdout import grade_decision
from evals.run_agent_v4_claim_holdout import request_payload


def test_all_20_cases_are_frozen_and_gold_is_not_model_visible():
    holdout, signatures = load_holdout()
    for case in holdout["cases"]:
        visible = public_case(case, signatures)
        assert "gold" not in visible
        payload = request_payload(visible, "deepseek-flash")
        assert "gold" not in payload["messages"][1]["content"]


def test_grader_rejects_unverified_citation_and_clinical_claim():
    holdout, _ = load_holdout()
    case = next(case for case in holdout["cases"] if case["id"] == "v4_12")
    decision = {**case["gold"], "clinical_use_supported": False,
                "citations": [{"pmid": "12538830", "scope": "candidate"}],
                "reason": "Only target engagement is established."}
    assert grade_decision(case, decision)["overall"]
    decision["citations"] = [{"pmid": "99999999", "scope": "candidate"}]
    assert not grade_decision(case, decision)["citation_fidelity"]
    decision["clinical_use_supported"] = True
    assert not grade_decision(case, decision)["clinical_abstention"]


def test_known_contradiction_requires_a_citation():
    holdout, _ = load_holdout()
    case = next(case for case in holdout["cases"] if case["id"] == "v4_02")
    decision = {**case["gold"], "clinical_use_supported": False,
                "citations": [{"pmid": "12538830", "scope": "candidate"}],
                "reason": "Target engagement is not efficacy."}
    assert not grade_decision(case, decision)["citation_coverage"]
    decision["citations"].append({"pmid": "24736075", "scope": "class"})
    assert grade_decision(case, decision)["overall"]
