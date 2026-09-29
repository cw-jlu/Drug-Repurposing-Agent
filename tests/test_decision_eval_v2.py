"""Invariant checks for the frozen hybrid decision-layer evaluation."""

import json

import pytest

from evals.build_decision_eval_v2 import NOTES
from evals.run_decision_eval_v1 import j0_decide
from evals.run_decision_eval_v2 import (
    RESULT, _score, load_frozen, note_payload, parse_note_response,
)


def _response(arguments, *, finish="tool_calls"):
    return {"choices": [{"finish_reason": finish, "message": {"tool_calls": [
        {"function": {"name": "classify_note", "arguments": json.dumps(arguments)}}
    ]}}]}


def test_frozen_cases_have_disjoint_notes_and_hidden_labels():
    config, _ = load_frozen()
    cases = config["cases"]
    assert len(cases) == 40
    assert len({case["state"]["notes"] for case in cases}) == 40
    assert {case["node_type"] for case in cases} == set(NOTES)
    assert sum(case["note_is_concern"] for case in cases) == 20
    for case in cases:
        assert case["reference_answer"] in case["options"]
        payload = note_payload(case["state"]["notes"])
        visible = json.dumps(payload, ensure_ascii=False)
        assert "reference_answer" not in visible
        assert "note_is_concern" not in visible
        assert payload["tools"][0]["function"]["strict"] is True


def test_note_parser_accepts_only_one_strict_label():
    assert parse_note_response(_response({"label": "CONCERN"})) == "CONCERN"
    for bad in ({"label": "YES"}, {"label": "CONCERN", "reason": "extra"}, {}):
        with pytest.raises(ValueError):
            parse_note_response(_response(bad))
    with pytest.raises(ValueError):
        parse_note_response(_response({"label": "CONCERN"}, finish="length"))


def test_saved_scores_recalculate_from_frozen_cases():
    config, _ = load_frozen()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    cases = config["cases"]
    for layer, field in (("j0_rules", "j0"), ("hybrid", "hybrid"),
                         ("full_llm", "full_llm")):
        predictions = {row["case_id"]: row[field] for row in result["case_results"]}
        assert _score(cases, predictions) == result["layers"][layer]
    assert [row["j0"] for row in result["case_results"]] == [
        j0_decide(case) for case in cases]
