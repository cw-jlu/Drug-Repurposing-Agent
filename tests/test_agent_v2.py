import json

from drug_repurposing_agent.agent_v2 import (
    MAX_ROUNDS, PlanV2, PlanningState, RulePlannerV2, Step, run_agent_v2, simulated_backend, tools_for,
    validate_plan,
)
from drug_repurposing_agent.workflow import Mode

OPEN, STRICT = Mode.RESEARCH_OPEN, Mode.BENCHMARK_STRICT
LUAD_INPUTS = ("disease_series", "drug_signatures")


def plan(*tools):
    return PlanV2("t", "r", tuple(Step(t) for t in tools))


def test_validate_dependencies_inputs_mode_and_terminal_review():
    assert validate_plan(plan("qc_disease_cohort", "differential_expression", "rank_candidates"), OPEN, LUAD_INPUTS, []) is None
    assert "earlier step" in validate_plan(plan("rank_candidates"), OPEN, LUAD_INPUTS, [])
    assert "unavailable inputs" in validate_plan(plan("qc_disease_cohort"), OPEN, ("drug_signatures",), [])
    assert "not allowed" in validate_plan(plan("qc_disease_cohort"), STRICT, LUAD_INPUTS, [])
    assert "last step" in validate_plan(plan("manual_review", "qc_disease_cohort"), OPEN, LUAD_INPUTS, [])
    assert "unknown tool" in validate_plan(plan("delete_everything"), OPEN, LUAD_INPUTS, [])
    assert validate_plan(plan("rank_candidates"), OPEN, LUAD_INPUTS, ["signature"]) is None


class Scripted:
    name = "scripted"

    def __init__(self, plans):
        self.plans, self.seen = list(plans), []

    def plan(self, state, tools):
        self.seen.append(json.loads(json.dumps(state.public())))
        return self.plans.pop(0)


def test_replan_after_failure_keeps_completed_outputs(tmp_path):
    first = plan("qc_disease_cohort", "differential_expression", "rank_candidates", "build_report")
    second = plan("rank_candidates", "build_report")
    planner = Scripted([first, second])
    out = run_agent_v2("q", OPEN, LUAD_INPUTS, planner, simulated_backend({"rank_candidates": 1}), tmp_path)
    assert out["status"] == "completed" and len(out["rounds"]) == 2
    assert planner.seen[1]["already_produced"] == ["cohort", "signature"]
    assert planner.seen[1]["observations"][-1]["tool"] == "rank_candidates"
    assert [e["tool"] for e in out["executed"] if e["status"] == "ok"] == [
        "qc_disease_cohort", "differential_expression", "rank_candidates", "build_report"]
    assert (tmp_path / "agent_v2_run.json").is_file()


def test_invalid_plans_exhaust_rounds_to_manual_review(tmp_path):
    planner = Scripted([plan("rank_candidates")] * MAX_ROUNDS)
    out = run_agent_v2("q", OPEN, LUAD_INPUTS, planner, simulated_backend(), tmp_path)
    assert out["status"] == "manual_review_required" and not out["executed"]


def test_rule_planner_full_partial_unsafe_and_benchmark():
    s = lambda q, mode=OPEN, inputs=LUAD_INPUTS: PlanningState(q, mode, inputs)
    tools = lambda mode: [t.public() for t in tools_for(mode)]
    full = RulePlannerV2().plan(s("请为肺腺癌筛选候选药物并给出证据报告"), tools(OPEN))
    assert [x.tool for x in full.steps][-1] == "build_report" and "review_literature" in [x.tool for x in full.steps]
    unsafe = RulePlannerV2().plan(s("给这位肺腺癌患者推荐用药剂量"), tools(OPEN))
    assert [x.tool for x in unsafe.steps] == ["manual_review"]
    bench = RulePlannerV2().plan(s("对 TRANSCRIPT 基准做排名", STRICT, ("items", "users")), tools(STRICT))
    assert [x.tool for x in bench.steps] == ["rank_transcriptome"]
