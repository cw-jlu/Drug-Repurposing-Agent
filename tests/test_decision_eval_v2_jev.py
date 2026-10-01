from evals.run_decision_eval_v2 import load_frozen
from evals.run_decision_eval_v2_jev import decide


def test_hybrid_requires_no_concern_and_gate_uses_v1_thresholds():
    config, _ = load_frozen()
    case = next(c for c in config["cases"] if c["risk_level"] == "high")
    note = {case["id"]: {"label": "NO_CONCERN", "confidence": 0.85}}
    out = decide([case], note, {})
    assert out["hybrid_jev_gated"][case["id"]] == case["review_option"]  # 0.85 < 0.9 for high risk
    concern = decide([case], {case["id"]: {"label": "CONCERN", "confidence": 0.99}}, {})
    assert concern["hybrid_jev"][case["id"]] == case["review_option"]
    assert decide([case], {}, {})["full_jev"][case["id"]] == case["review_option"]  # failed call fails closed
