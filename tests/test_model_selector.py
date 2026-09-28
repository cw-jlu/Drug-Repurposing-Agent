"""Method selection must be prospective, bounded, and score-blind."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from drug_repurposing_agent.model_selector import select_methods


CONFIG = Path(__file__).resolve().parents[1] / "configs/component_selector_v1.json"


def response(method: str = "B1k") -> dict:
    choices = {split: {"method": method, "reason": "Hypothesis, not observed performance."}
               for split in ("random_simple", "weakly_correlated")}
    return {"choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
        {"function": {"name": "submit_method_choice", "arguments": json.dumps(choices)}}
    ]}}]}


def test_selector_sees_descriptions_but_no_test_scores():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    seen = {}
    def transport(payload):
        seen.update(payload)
        return response()
    result = select_methods(config, transport)
    assert result["choices"]["random_simple"]["method"] == "B1k"
    assert seen["tools"][0]["function"]["strict"] is True
    assert "NS-AUC" not in json.dumps(seen)
    assert "test_score" not in json.dumps(seen)


def test_rejects_out_of_set_method():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Unknown method"):
        select_methods(config, lambda _: response("BNNR"))


def test_rejects_test_metric_in_input():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["dataset"]["NS-AUC"] = 0.99
    with pytest.raises(ValueError, match="forbidden metrics"):
        select_methods(config, lambda _: response())
