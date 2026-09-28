import json
from pathlib import Path

import pytest

from drug_repurposing_agent.agent import AgentInputs, PlanningContext, StructuredPlanner, TOOLS, run_agent_task
from drug_repurposing_agent.deepseek import DeepSeekPlanner, DeepSeekPlannerError
from drug_repurposing_agent.trace import TraceRecorder, redact, traced_run, verify_trace_chain
from drug_repurposing_agent.workflow import Mode, run_expression_workflow
from evals.run_planner_eval import run_one


def events(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_trace_is_durable_and_redacts_credentials_and_internal_reasoning(tmp_path):
    trace = TraceRecorder("test", tmp_path)
    trace.emit("request", prompt="Use sk-abcdefghijklmnop and Bearer abc123",
               authorization="Bearer abc123", nested={"api_key": "secret",
                                                       "reasoning_content": "private thought"})
    data = trace.path.read_text(encoding="utf-8")
    assert len(events(trace.path)) == 2
    for secret in ("sk-abcdefghijklmnop", "Bearer abc123", "private thought", "secret"):
        assert secret not in data
    assert events(trace.path)[-1]["nested"]["api_key"] == "[REDACTED]"
    assert redact("DEEPSEEK_API_KEY=abc123") == "DEEPSEEK_API_KEY=[REDACTED]"
    verified, integrity = verify_trace_chain(trace.path, require_chain=True)
    assert integrity == "sha256_chain_v1"
    assert verified[-1]["sequence"] == 2


def test_trace_chain_rejects_reordered_or_modified_events(tmp_path):
    trace = TraceRecorder("chain", tmp_path)
    trace.emit("one", value=1)
    trace.emit("two", value=2)
    original = trace.path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in original.splitlines()]
    events[1]["value"] = 99
    trace.path.write_text("\n".join(json.dumps(event) for event in events) + "\n",
                          encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        verify_trace_chain(trace.path, require_chain=True)


def test_deepseek_trace_records_request_response_selection_and_bad_call(tmp_path, monkeypatch):
    monkeypatch.setenv("DRUG_AGENT_TRACE_DIR", str(tmp_path))
    tools = tuple(spec.public_schema() for spec in TOOLS.values())
    context = PlanningContext(Mode.RESEARCH_OPEN, ("items", "users"))
    good = DeepSeekPlanner("sk-abcdefghijklmnop", transport=lambda _: {
        "choices": [{"message": {"tool_calls": [{"function": {
            "name": "rank_transcriptome", "arguments": '{"top_k": 5}'}}]}}],
        "reasoning_content": "private thought", "usage": {"total_tokens": 12},
    })
    good.plan("rank sk-abcdefghijklmnop", context, tools)
    stages = [event["stage"] for event in events(good.last_trace_path)]
    assert stages == ["trace_started", "model_request", "model_response", "tool_call_selected"]
    text = Path(good.last_trace_path).read_text(encoding="utf-8")
    assert "sk-abcdefghijklmnop" not in text
    assert "private thought" not in text
    assert good.last_metadata["trace_file"] == good.last_trace_path

    bad = DeepSeekPlanner("test-key", transport=lambda _: {"choices": []})
    with pytest.raises(DeepSeekPlannerError):
        bad.plan("rank", context, tools)
    assert events(bad.last_trace_path)[-1]["stage"] == "tool_call_invalid"


def test_agent_and_eval_keep_failure_traces(tmp_path):
    def fail(_):
        raise RuntimeError("provider unavailable")
    planner = StructuredPlanner(fail)
    report = run_agent_task("rank", AgentInputs(), tmp_path / "agent", planner=planner)
    assert report["status"] == "blocked"
    assert events(report["trace_file"])[-1]["stage"] == "plan_blocked"
    assert Path(report["trace_file"]).exists()

    trace = TraceRecorder("eval", tmp_path / "eval")
    case = {"id": "c1", "mode": "research_open", "available_inputs": [],
            "question": "rank", "expected_tool": "manual_review"}
    record = run_one(planner, case, trace)
    assert record["actual_tool"] == "planner_error"
    assert events(trace.path)[-1]["stage"] == "case_finished"
    assert events(trace.path)[-1]["error_type"] == "RuntimeError"


def test_workflow_failure_is_recorded(tmp_path):
    output = tmp_path / "workflow"
    with pytest.raises(ValueError, match="top_k"):
        run_expression_workflow(tmp_path / "missing1", tmp_path / "missing2", output, top_k=0)
    traces = list((output / "traces").glob("*.jsonl"))
    assert len(traces) == 1
    assert events(traces[0])[-1]["stage"] == "workflow_failed"


def test_traced_entrypoint_keeps_failure_without_result_file(tmp_path):
    def fail(trace):
        trace.emit("checked_input", sha256="abc")
        raise FileNotFoundError("expected input missing")

    with pytest.raises(FileNotFoundError):
        traced_run("benchmark", fail, tmp_path)
    traces = list(tmp_path.glob("*.jsonl"))
    assert len(traces) == 1
    assert [event["stage"] for event in events(traces[0])] == [
        "trace_started", "checked_input", "run_failed"]
