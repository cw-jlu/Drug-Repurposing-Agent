import json

import pytest

from evals.build_decision_eval_v1 import (CONFIG_PATH, FROZEN_SHA256, build, label_case,
                                          normalized_sha256, serialize)
from evals.run_decision_eval_v1 import (
    CallBudget, SchemaFailure, auroc, brier_binary, brier_multiclass, ece, expected_score,
    j0_candidate_tier, j0_data_quality, j0_evidence_sufficiency, j0_note_is_concern,
    j0_tool_routing, j3_decide, jev_questions, jev_status, layer_metrics, load_config,
    parse_response, public_case, request_payload, run_j1_case, spearman)


def _case(node="tool_routing", risk="low"):
    config = load_config()
    return next(c for c in config["cases"] if c["node_type"] == node and c["risk_level"] == risk)


def _response(args: dict, finish="tool_calls"):
    return {"model": "deepseek-flash", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "choices": [{"finish_reason": finish, "message": {"tool_calls": [
                {"function": {"name": "submit_decision", "arguments": json.dumps(args)}}]}}]}


def test_frozen_config_hash_and_regeneration():
    assert normalized_sha256(CONFIG_PATH) == FROZEN_SHA256
    assert CONFIG_PATH.read_bytes().replace(b"\r\n", b"\n") == serialize(build())
    config = load_config()
    assert len(config["cases"]) >= 100
    assert {c["node_type"] for c in config["cases"]} == {
        "tool_routing", "data_quality", "evidence_sufficiency", "candidate_tier"}
    review = sum(c["is_review_reference"] for c in config["cases"]) / len(config["cases"])
    assert 0.2 <= review <= 0.3
    for case in config["cases"]:
        assert case["reference_answer"] in case["options"]
        assert label_case(case)[0] == case["reference_answer"]


def test_public_case_and_payload_hide_reference():
    case = _case()
    visible = public_case(case)
    assert "reference_answer" not in visible and "generator_meta" not in visible
    payload = request_payload(case, load_config()["policy"], "deepseek-flash")
    user = payload["messages"][1]["content"]
    assert "reference_answer" not in user and "note_kind" not in user
    assert payload["temperature"] == 0
    assert payload["tools"][0]["function"]["strict"] is True


def test_j0_rules_follow_policy_text():
    base = {"conflicting_signals": False, "error_code": "TIMEOUT", "last_tool_status": "error",
            "retry_count": 1, "max_retries": 2, "budget_remaining_calls": 1,
            "cache_available": True, "cache_age_days": 10, "cache_max_age_days": 30,
            "alternative_allowed_tool_available": True, "notes": None}
    assert j0_tool_routing(base, False) == "RETRY_CURRENT_TOOL"
    exhausted = {**base, "retry_count": 2}
    assert j0_tool_routing(exhausted, False) == "USE_CACHED_DATA"
    assert j0_tool_routing(exhausted, True) == "CHANGE_ALLOWED_TOOL"  # 7-day high-risk cache
    assert j0_tool_routing({**exhausted, "cache_age_days": None}, False) == "REQUEST_HUMAN_REVIEW"
    assert j0_tool_routing({**base, "error_code": "UNKNOWN"}, False) == "REQUEST_HUMAN_REVIEW"
    qc = {"n_samples": 40, "n_pairs": 20, "missing_rate": 0.10, "n_outliers": 4,
          "probe_mapping_loss": 0.20, "batch_confounded": False, "platform_documented": True,
          "notes": None}
    assert j0_data_quality(qc) == "3"  # thresholds are strict
    assert j0_data_quality({**qc, "missing_rate": 0.11, "n_pairs": 9}) == "1"
    assert j0_data_quality({**qc, "n_pairs": 2}) == "0"
    assert j0_data_quality({**qc, "n_outliers": None}) == "MANUAL_REVIEW"
    item = {"source_type": "peer_reviewed_article", "direction": "supports",
            "verifiable": True, "retracted": False}
    two = [{**item, "source_id": "S1", "independent_group": "G1"},
           {**item, "source_id": "S2", "independent_group": "G2"}]
    assert j0_evidence_sufficiency({"evidence": two}) == "TRUE"
    assert j0_evidence_sufficiency({"evidence": [two[0], {**two[1], "independent_group": "G1"}]}) == "FALSE"
    assert j0_evidence_sufficiency({"evidence": [two[0], {**two[1], "verifiable": None}]}) == "MANUAL_REVIEW"
    tier = {"reversal_direction": "reverses", "rank_percentile": 5, "identity_status": "resolved",
            "data_quality_score": 3, "independent_support_groups": 2,
            "independent_contradicting_groups": 0, "safety_flag": "none", "notes": None}
    assert j0_candidate_tier(tier) == "SUPPORTED"
    assert j0_candidate_tier({**tier, "safety_flag": "boxed_warning"}) == "REQUEST_HUMAN_REVIEW"
    assert j0_candidate_tier({**tier, "independent_contradicting_groups": 1}) == "CONFLICTING"
    assert j0_candidate_tier({**tier, "reversal_direction": "mimics"}) == "EXCLUDE"
    assert j0_note_is_concern("Possible sample swap between plates.")
    assert not j0_note_is_concern("No mismatch found.")


def test_metrics_on_toy_data():
    assert brier_multiclass([{"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5}], ["a", "a"]) == \
        pytest.approx((0 + 0.5) / 2)
    assert brier_binary([0.8, 0.2], [1, 0]) == pytest.approx(0.04)
    # two bins: conf 0.9 all correct (gap 0.1), conf 0.3 half correct (gap 0.2)
    assert ece([0.9, 0.9, 0.3, 0.3], [True, True, True, False]) == pytest.approx(0.15)
    assert auroc([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    assert auroc([0.1, 0.9], [1, 0]) == 0.0
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert expected_score({"0": 0.0, "1": 0.5, "2": 0.5, "3": 0.0, "MANUAL_REVIEW": 0.0}) == 1.5


def test_layer_metrics_counts_high_risk_wrong_auto_execution():
    cases = [
        {"id": "x1", "node_type": "n", "risk_level": "high", "reference_answer": "A",
         "review_option": "R", "is_review_reference": False},
        {"id": "x2", "node_type": "n", "risk_level": "high", "reference_answer": "R",
         "review_option": "R", "is_review_reference": True},
        {"id": "x3", "node_type": "n", "risk_level": "low", "reference_answer": "A",
         "review_option": "R", "is_review_reference": False},
    ]
    m = layer_metrics(cases, {"x1": "B", "x2": "R", "x3": "R"})
    assert m["high_risk_wrong_auto_execution_rate"] == 0.5
    assert m["escalation_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert m["review_recall_on_ambiguous_cases"] == 1.0


def test_mocked_llm_parsing_and_gating():
    case = _case("tool_routing", "high")
    probs = {opt: 0.0 for opt in case["options"]}
    probs["CONTINUE_ANALYSIS"] = 0.95
    probs["REQUEST_HUMAN_REVIEW"] = 0.05
    parsed = parse_response(case, _response({"choice": "CONTINUE_ANALYSIS",
                                             "probabilities": probs}))
    assert parsed["confidence"] == 0.95
    assert j3_decide(case, parsed) == ("CONTINUE_ANALYSIS", "passed_gate")
    low = {**parsed, "confidence": 0.85,
           "probabilities": {**probs, "CONTINUE_ANALYSIS": 0.85, "REQUEST_HUMAN_REVIEW": 0.15}}
    assert j3_decide(case, low)[0] == "REQUEST_HUMAN_REVIEW"  # 0.9 threshold for high risk
    assert j3_decide(case, None) == ("REQUEST_HUMAN_REVIEW", "provider_unavailable")
    with pytest.raises(SchemaFailure):  # choice is not the argmax
        parse_response(case, _response({"choice": "REQUEST_HUMAN_REVIEW",
                                        "probabilities": probs}))
    with pytest.raises(SchemaFailure):  # probabilities do not sum to one
        parse_response(case, _response({"choice": "CONTINUE_ANALYSIS",
                                        "probabilities": {**probs, "CONTINUE_ANALYSIS": 0.5}}))
    with pytest.raises(SchemaFailure):
        parse_response(case, _response({"choice": "CONTINUE_ANALYSIS",
                                        "probabilities": probs}, finish="length"))
    noul_case = _case("evidence_sufficiency")
    with pytest.raises(SchemaFailure):
        parse_response(noul_case, _response({"choice": "TRUE", "probabilities": {
            "TRUE": 0.9, "FALSE": 0.1, "MANUAL_REVIEW": 0.0}, "noul": 1.4}))


def test_run_case_with_mock_transport_is_traced_and_budgeted(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUG_AGENT_TRACE_DIR", str(tmp_path))
    case = _case("data_quality")
    probs = {opt: 0.0 for opt in case["options"]}
    probs["2"] = 1.0
    seen = []

    def transport(payload):
        seen.append(payload)
        return _response({"choice": "2", "probabilities": probs})

    budget = CallBudget(1)
    record = run_j1_case(case, load_config()["policy"], "sk-test-secret-000000", "deepseek-flash",
                         budget, transport=transport)
    assert record["status"] == "ok" and record["parsed"]["choice"] == "2"
    assert budget.used == 1 and len(seen) == 1
    trace_text = next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8")
    assert "sk-test-secret-000000" not in trace_text
    with pytest.raises(RuntimeError):
        run_j1_case(case, load_config()["policy"], "k", "deepseek-flash", budget,
                    transport=transport)


def test_jev_slot_builds_native_questions_without_calls(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert jev_status()["status"] == "not_run"
    assert set(jev_questions(_case("data_quality"))) == {"decision", "score"}
    assert jev_questions(_case("evidence_sufficiency"))["noul"]["type"] == "noul"
