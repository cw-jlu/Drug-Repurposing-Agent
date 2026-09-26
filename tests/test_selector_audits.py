import json
from pathlib import Path

import numpy as np
import pytest

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.deepseek import DeepSeekPlanner
from drug_repurposing_agent.model_selector import select_partition_method
from drug_repurposing_agent.trace import TraceRecorder
from evals.audit_historical_selector_v1 import audit, paired_summary
from evals.grade_method_selection_traces import grade_choices


def test_paired_summary_counts_actual_seed_differences():
    result = paired_summary(np.array([0.6, 0.5, 0.4]), np.array([0.5, 0.5, 0.6]))
    assert result["paired_mean_delta"] == pytest.approx(-1 / 30)
    assert (result["seed_wins"], result["seed_ties"], result["seed_losses"]) == (1, 1, 1)


def test_historical_selector_audit_uses_frozen_official_files():
    result = audit(Path("benchmark/results/recess_component_llm_selector_v1.json"),
                   Path("configs/component_selector_v1.json"), Path("benchmark/results"))
    random = result["splits"]["random_simple"]
    weak = result["splits"]["weakly_correlated"]
    assert random["selected_method"] == "B2"
    assert random["selected_vs_b2"]["paired_mean_delta"] == 0
    assert weak["selected_method"] == "B1"
    assert weak["selected_vs_b2"]["paired_mean_delta"] == pytest.approx(0.0398254922)
    assert weak["selected_vs_b2"]["seed_wins"] == 100


def make_choice_fixture(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DRUG_AGENT_TRACE_DIR", str(tmp_path / "provider"))
    blind = json.loads(Path("configs/method_selection_partition_v2_cases.json")
                       .read_text(encoding="utf-8"))["cases"][0]["blind_input"]
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"partition_count": 1,
                                         "model_repeats_per_partition": 1}), encoding="utf-8")
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps({"protocol_sha256": sha256_file(protocol_path),
                                      "cases": [{"id": "case1", "blind_input": blind}]}),
                          encoding="utf-8")
    main_trace = TraceRecorder("method_selection_v2", tmp_path / "main")
    main_trace.emit("selection_started", cases=str(cases_path), protocol=str(protocol_path))
    provider = DeepSeekPlanner("dummy", transport=lambda _: {
        "model": "deepseek-flash", "choices": [{"message": {"tool_calls": [{"function": {
            "name": "submit_partition_choice",
            "arguments": json.dumps({"method": "B1", "reason": "Feature-only choice"})}}]}}],
        "usage": {"total_tokens": 9},
    })
    response = select_partition_method(blind, provider._post)
    row = {"case_id": "case1", "repeat": 1, **response,
           "provider_trace_file": provider.last_trace_path,
           "provider_trace_sha256": sha256_file(Path(provider.last_trace_path))}
    main_trace.emit("choice_validated", **row)
    report_path = tmp_path / "choices.json"
    report_path.write_text(json.dumps({"status": "prescore_choices_frozen_outcomes_unseen",
                                       "protocol_sha256": sha256_file(protocol_path),
                                       "cases_sha256": sha256_file(cases_path),
                                       "model": "deepseek-flash", "choices": [row],
                                       "trace_file": str(main_trace.path)}), encoding="utf-8")
    main_trace.emit("selection_saved", output=str(report_path),
                    output_sha256=sha256_file(report_path), choices=1)
    return report_path, cases_path, protocol_path, Path(provider.last_trace_path)


def test_method_choice_trace_grade_reconciles_request_and_response(tmp_path, monkeypatch):
    report, cases, protocol, _ = make_choice_fixture(tmp_path, monkeypatch)
    grade = grade_choices(report, cases, protocol)
    assert grade["choices"] == grade["passed"] == 1
    assert grade["rows"][0]["blind_request_match"]
    assert grade["rows"][0]["provider_choice_match"]


def test_method_choice_trace_grade_catches_provider_response_change(tmp_path, monkeypatch):
    report, cases, protocol, provider_path = make_choice_fixture(tmp_path, monkeypatch)
    events = [json.loads(line) for line in provider_path.read_text(encoding="utf-8").splitlines()]
    events[-1]["response"]["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"] = json.dumps({"method": "B2", "reason": "Altered"})
    provider_path.write_text("\n".join(json.dumps(event) for event in events) + "\n",
                             encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        grade_choices(report, cases, protocol)
