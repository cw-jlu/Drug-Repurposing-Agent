import json
from pathlib import Path

import pytest

from drug_repurposing_agent.model_selector import select_partition_method
from evals.score_method_selection_partition_v2 import score_choices


def blind_case():
    path = Path("configs/method_selection_partition_v2_cases.json")
    return json.loads(path.read_text(encoding="utf-8"))["cases"][0]["blind_input"]


def test_frozen_partition_case_is_label_blind():
    case = blind_case()
    assert set(case) == {"dataset", "partition_id", "drug_count", "disease_count",
                         "expression_gene_count", "median_disease_signature_sd", "split",
                         "split_description", "methods", "metric_to_optimize"}
    text = json.dumps(case).lower()
    assert "positive_pairs" not in text
    assert "heldout_auc" not in text
    assert "ratings" not in text


def test_partition_choice_uses_only_blind_input_and_validates_method():
    seen = {}
    def fake(payload):
        seen.update(payload)
        return {"choices": [{"message": {"tool_calls": [{"function": {
            "name": "submit_partition_choice", "arguments": json.dumps({
                "method": "B1", "reason": "Feature-only signal; uncertain performance"})}}]}}],
                "usage": {"total_tokens": 17}}
    result = select_partition_method(blind_case(), fake)
    assert result["choice"]["method"] == "B1"
    assert result["usage"]["total_tokens"] == 17
    assert seen["thinking"] == {"type": "disabled"}
    with pytest.raises(ValueError, match="Unexpected"):
        select_partition_method({**blind_case(), "test_auc": 0.9}, fake)


def test_score_choices_averages_all_repeats_without_best_of_selection():
    cases = [{"id": "one"}, {"id": "two"}]
    outcomes = {"one": {"B0p": 0.5, "B1k": 0.4, "B1": 0.6, "B2": 0.55},
                "two": {"B0p": 0.5, "B1k": 0.3, "B1": 0.45, "B2": 0.52}}
    choices = [{"case_id": "one", "repeat": 1, "choice": {"method": "B1"}},
               {"case_id": "one", "repeat": 2, "choice": {"method": "B2"}},
               {"case_id": "two", "repeat": 1, "choice": {"method": "B1"}},
               {"case_id": "two", "repeat": 2, "choice": {"method": "B2"}}]
    result = score_choices(choices, cases, outcomes, 2)
    assert result["by_case"]["one"]["mean_selected_ns_auc"] == pytest.approx(0.575)
    assert result["by_case"]["two"]["mean_selected_ns_auc"] == pytest.approx(0.485)
    assert result["mean_selected_ns_auc_across_cases"] == pytest.approx(0.53)
    with pytest.raises(ValueError, match="exactly once"):
        score_choices(choices[:-1], cases, outcomes, 2)
