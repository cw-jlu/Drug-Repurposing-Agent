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


def test_fetch_step_satisfies_disease_series_and_registered_disease_is_required():
    assert validate_plan(plan("fetch_geo_series", "qc_disease_cohort"), OPEN, ("registered_disease",), []) is None
    assert "unavailable inputs" in validate_plan(plan("fetch_geo_series"), OPEN, ("disease_series",), [])
    assert "not allowed" in validate_plan(plan("fetch_geo_series"), STRICT, ("registered_disease",), [])


def test_rule_planner_fetches_a_registered_disease_that_is_not_downloaded():
    tools = [t.public() for t in tools_for(OPEN)]
    state = PlanningState("请为肺腺癌筛选候选药物", OPEN, ("registered_disease", "drug_signatures"))
    steps = [s.tool for s in RulePlannerV2().plan(state, tools).steps]
    assert steps[:2] == ["fetch_geo_series", "qc_disease_cohort"] and steps[-1] == "build_report"
    on_disk = PlanningState("请为肺腺癌筛选候选药物", OPEN, ("registered_disease", "disease_series", "drug_signatures"))
    assert "fetch_geo_series" not in [s.tool for s in RulePlannerV2().plan(on_disk, tools).steps]


def test_available_inputs_depend_on_the_registered_disease_in_the_question():
    from drug_repurposing_agent.luad_tools_v2 import available_inputs
    other = available_inputs("请为乳腺癌筛选候选药物")
    assert "registered_disease" not in other and "disease_series" not in other
    assert "registered_disease" in available_inputs("请为肺腺癌筛选候选药物")


def test_fetch_tool_is_offered_only_when_a_registered_disease_is_named():
    names = lambda inputs: {t.name for t in tools_for(OPEN, inputs)}
    assert "fetch_geo_series" not in names(("disease_series", "drug_signatures"))
    assert "fetch_geo_series" in names(("registered_disease",))
    assert "fetch_geo_series" in {t.name for t in tools_for(OPEN)}
