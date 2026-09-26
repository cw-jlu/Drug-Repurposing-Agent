import json
from pathlib import Path

import pytest
import numpy as np

from drug_repurposing_agent.model_selector import select_partition_method
from drug_repurposing_agent.deepseek import DeepSeekPlanner
from evals.score_method_selection_partition_v2 import score_choices
from scripts.run_method_selection_partition_v2 import recover_attempt_rows, recover_first_choice
from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder


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


def test_recover_first_long_reason_from_original_provider_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUG_AGENT_TRACE_DIR", str(tmp_path))
    reason = "Evidence remains uncertain. " * 26
    planner = DeepSeekPlanner("dummy", transport=lambda _: {
        "choices": [{"message": {"tool_calls": [{"function": {
            "name": "submit_partition_choice",
            "arguments": json.dumps({"method": "B2", "reason": reason})}}]}}]})
    first = select_partition_method(blind_case(), planner._post)
    recovered = recover_first_choice(Path(planner.last_trace_path), blind_case(), "deepseek-flash")
    assert recovered == first
    assert 200 < len(recovered["choice"]["reason"]) < 1000
    with pytest.raises(ValueError, match="differs"):
        recover_first_choice(Path(planner.last_trace_path),
                             {**blind_case(), "disease_count": 999}, "deepseek-flash")


def test_recover_failed_attempt_preserves_every_provider_response(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUG_AGENT_TRACE_DIR", str(tmp_path))
    cases = json.loads(Path("configs/method_selection_partition_v2_cases.json")
                       .read_text(encoding="utf-8"))["cases"]
    reasons = ["short reason", "uncertain " * 120]
    paths = []
    responses = []
    for reason in reasons:
        planner = DeepSeekPlanner("dummy", transport=lambda _, text=reason: {
            "choices": [{"message": {"tool_calls": [{"function": {
                "name": "submit_partition_choice",
                "arguments": json.dumps({"method": "B1", "reason": text})}}]}}]})
        responses.append(select_partition_method(cases[0]["blind_input"], planner._post))
        paths.append(Path(planner.last_trace_path))
    attempt = TraceRecorder("failed_selection", tmp_path)
    attempt.emit("selection_started")
    attempt.emit("choice_validated", case_id=cases[0]["id"], repeat=1,
                 **responses[0], provider_trace_file=str(paths[0]),
                 provider_trace_sha256=sha256_file(paths[0]))
    attempt.emit("selection_failed", error="Invalid partition choice or reason",
                 provider_trace=str(paths[1]))
    rows = recover_attempt_rows(attempt.path, cases, 3, "deepseek-flash")
    assert [(row["case_id"], row["repeat"]) for row in rows] == [
        (cases[0]["id"], 1), (cases[0]["id"], 2)]
    assert [row["choice"] for row in rows] == [item["choice"] for item in responses]
    assert all(row["recovery_attempt_trace_sha256"] == sha256_file(attempt.path)
               for row in rows)


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


def test_score_choices_uses_paired_seed_deltas_without_best_of_repeats():
    cases = [{"id": "one"}]
    vectors = {"one": {"B0p": np.array([0.5, 0.5]), "B1k": np.array([0.4, 0.4]),
                       "B1": np.array([0.6, 0.4]), "B2": np.array([0.5, 0.5])}}
    outcomes = {"one": {method: float(values.mean())
                        for method, values in vectors["one"].items()}}
    choices = [{"case_id": "one", "repeat": 1, "choice": {"method": "B1"}},
               {"case_id": "one", "repeat": 2, "choice": {"method": "B2"}}]
    report = score_choices(choices, cases, outcomes, 2, vectors)
    paired = report["by_case"]["one"]["paired_seed_delta_vs_b2"]
    assert paired["mean"] == pytest.approx(0)
    assert (paired["seed_wins"], paired["seed_ties"], paired["seed_losses"]) == (1, 0, 1)
    assert report["paired_partition_summary"]["tied_partitions"] == 1
