from evals.run_planner_eval_multistep import grade, load_cases


def test_frozen_cases_load_and_cover_categories():
    cases = load_cases()
    assert len(cases) == 40
    assert {c["category"] for c in cases} == {"full", "partial", "benchmark", "missing_input", "failure", "unsafe"}


def test_grade_checks_status_required_forbidden_and_order():
    case = {"expected": {"status": ["completed"], "must_succeed": ["a", "b"], "must_not_call": ["x"],
                         "order": [["a", "b"]]}}
    good = {"status": "completed", "rounds": [{}],
            "executed": [{"tool": "a", "status": "ok"}, {"tool": "b", "status": "ok"}]}
    assert grade(case, good)["passed"]
    bad = {"status": "completed", "rounds": [{}],
           "executed": [{"tool": "b", "status": "ok"}, {"tool": "x", "status": "failed"}, {"tool": "a", "status": "ok"}]}
    problems = grade(case, bad)["problems"]
    assert any("forbidden x" in p for p in problems) and any("order" in p for p in problems)
