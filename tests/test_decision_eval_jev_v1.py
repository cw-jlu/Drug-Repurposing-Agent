from evals.run_decision_eval_jev_v1 import jev_state, layers
from evals.run_decision_eval_v1 import load_config


def _case(cid, risk="low"):
    return {"id": cid, "node_type": "tool_routing", "risk_level": risk, "question": "Q?",
            "options": ["GO", "REQUEST_HUMAN_REVIEW"], "review_option": "REQUEST_HUMAN_REVIEW",
            "state": {"x": 1}, "reference_answer": "GO", "is_review_reference": False}


def _jev(choice, conf):
    probs = {"GO": 0.9, "REQUEST_HUMAN_REVIEW": 0.1} if choice == "GO" else {"GO": 0.1, "REQUEST_HUMAN_REVIEW": 0.9}
    return {"status": "ok", "answers": {"decision": {"type": "choice", "choice": choice,
                                                     "confidence": conf, "probabilities": probs}}}


def test_gate_and_llm_fallback():
    cases = [_case("c1"), _case("c2"), _case("c3", risk="high")]
    jev = {"c1": _jev("GO", 0.95), "c2": _jev("GO", 0.5), "c3": _jev("GO", 0.85)}
    flash = {"c2": {"parsed": {"choice": "GO", "confidence": 0.95,
                               "probabilities": {"GO": 0.95, "REQUEST_HUMAN_REVIEW": 0.05}}}}
    d, reasons = layers(cases, jev, flash)
    assert d["J2_jev"] == {"c1": "GO", "c2": "GO", "c3": "GO"}
    assert d["J3_jev_gate"]["c2"] == "REQUEST_HUMAN_REVIEW" and d["J3_jev_gate"]["c3"] == "REQUEST_HUMAN_REVIEW"
    assert d["J4_jev_gate_llm_fallback"]["c2"] == "GO"           # confident flash answer
    assert d["J4_jev_gate_llm_fallback"]["c3"] == "REQUEST_HUMAN_REVIEW"  # no flash answer -> fail closed
    assert reasons["c1"] == "passed_gate"


def test_state_never_contains_reference_fields():
    config = load_config()
    state = jev_state(config["cases"][0], config["policy"])
    text = str(state)
    assert "reference_answer" not in text and "generator_meta" not in text and "policy_rule" not in text
