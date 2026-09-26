import json

import pytest

from drug_repurposing_agent.agent import PlanningContext, TOOLS
from drug_repurposing_agent.deepseek import (
    DeepSeekConfig,
    DeepSeekPlanner,
    DeepSeekPlannerError,
    local_api_key,
)
from drug_repurposing_agent.workflow import Mode


def schemas(mode=Mode.RESEARCH_OPEN):
    return tuple(spec.public_schema() for spec in TOOLS.values() if mode in spec.modes)


def test_deepseek_planner_uses_required_tool_call_without_exposing_paths():
    captured = {}

    def fake_transport(payload):
        captured.update(payload)
        return {
            "model": "deepseek-flash",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {"tool_calls": [{"function": {
                    "name": "rank_transcriptome", "arguments": json.dumps({"top_k": 50})
                }}]},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    planner = DeepSeekPlanner("test-key", DeepSeekConfig(), fake_transport)
    plan = planner.plan(
        "筛选候选药物",
        PlanningContext(Mode.RESEARCH_OPEN, ("items", "users")),
        schemas(),
    )
    assert plan.calls[0].name == "rank_transcriptome"
    assert plan.calls[0].arguments == {"top_k": 50}
    assert captured["tool_choice"] == "required"
    assert captured["thinking"] == {"type": "disabled"}
    assert "test-key" not in json.dumps(captured)
    assert planner.last_metadata["usage"]["total_tokens"] == 15


def test_deepseek_planner_rejects_invalid_tool_arguments():
    def fake_transport(_):
        return {"choices": [{"message": {"tool_calls": [{"function": {
            "name": "rank_transcriptome", "arguments": "not-json"
        }}]}}]}

    planner = DeepSeekPlanner("test-key", transport=fake_transport)
    with pytest.raises(DeepSeekPlannerError, match="invalid tool call"):
        planner.plan("筛选候选药物", PlanningContext(Mode.RESEARCH_OPEN, ()), schemas())


def test_deepseek_planner_fails_closed_when_provider_invents_unavailable_tool():
    def fake_transport(_):
        return {"choices": [{"message": {"tool_calls": [{"function": {
            "name": "package_luad_case", "arguments": "{}"
        }}]}}]}

    planner = DeepSeekPlanner("test-key", transport=fake_transport)
    plan = planner.plan("LUAD", PlanningContext(Mode.BENCHMARK_STRICT, ()),
                        schemas(Mode.BENCHMARK_STRICT))
    assert plan.calls[0].name == "manual_review"
    assert planner.last_metadata["local_fallback"] == "unavailable_tool_to_manual_review"


def test_local_key_reads_ignored_env_without_printing_or_tracing_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=local-test-credential\n", encoding="utf-8")
    assert local_api_key() == "local-test-credential"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-test-credential")
    assert local_api_key() == "process-test-credential"
