import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import demo_data as dd  # noqa: E402
from drug_repurposing_agent.trace import verify_trace_chain  # noqa: E402


def test_importing_demo_does_not_start_ui():
    import demo  # noqa: F401  (must not require a running Streamlit server)
    assert callable(demo.main)


def test_official_comparison_has_twelve_models_per_split():
    summary = dd.load_official_comparison()
    ns = summary[summary.metric == "NS-AUC"]
    assert ns.groupby("split").model.nunique().to_dict() == {
        "random_simple": 12, "weakly_correlated": 12}
    b2 = ns[(ns.model == "B2") & (ns.split == "random_simple")].iloc[0]
    assert b2["mean"] == pytest.approx(0.5222, abs=1e-4)


def test_b2_csvs_reproduce_official_json_summary():
    per_seed = dd.load_result_dir(dd.B2_RESULT_DIR)
    local = dd.summarize_per_seed(per_seed, "b2_csv").set_index(["split", "model", "metric"])
    official = dd.load_official_comparison().set_index(["split", "model", "metric"])
    for split in dd.SPLITS:
        key = (split, "B2", "NS-AUC")
        assert local.loc[key, "n"] == 100
        assert local.loc[key, "mean"] == pytest.approx(official.loc[key, "mean"], abs=1e-9)
        assert local.loc[key, "sd"] == pytest.approx(official.loc[key, "sd"], abs=1e-9)


def test_missing_extra_dir_is_a_note_not_an_error(tmp_path):
    summary, per_seed, notes = dd.load_benchmark(extra_dirs=[tmp_path / "recess_official_b3"])
    assert set(summary.model) >= {"B2", "BNNR"} and len(set(summary.model)) == 12
    assert set(per_seed.model) == {"B2"}
    assert any("recess_official_b3" in note for note in notes)


def test_extra_dir_in_b2_format_adds_new_model(tmp_path):
    extra = tmp_path / "recess_official_b3"
    extra.mkdir()
    for split, values in {"random_simple": (0.6, 0.7), "weakly_correlated": (0.55, 0.65)}.items():
        name = f"results_N=2_B3_TRANSCRIPT_{split}_AUC_1.000000_5_0.200000.csv"
        (extra / name).write_text(
            ",1_B3,2_B3\nAUC,0.5,0.5\nLin's AUC,%s,%s\nglobal AUC,0.9,0.8\n" % values,
            encoding="utf-8")
    summary, per_seed, notes = dd.load_benchmark(extra_dirs=[extra])
    b3 = summary[(summary.model == "B3") & (summary.metric == "NS-AUC")].set_index("split")
    assert b3.loc["random_simple", "mean"] == pytest.approx(0.65)
    assert b3.loc["weakly_correlated", "n"] == 2
    assert set(per_seed.model) == {"B2", "B3"}
    table = dd.ranking_table(summary)
    assert len(table) == 13 and "B3" in set(table["模型"])
    assert any("B3" in note for note in notes)


def test_luad_candidates_have_evidence_cards_and_pmid_links():
    frame, config = dd.load_luad_candidates()
    assert list(frame["rank"]) == list(range(1, 11))
    assert set(frame.confidence) == {"insufficient_evidence"}
    assert frame.loc[0, "name"] == "hydrocortisone"
    assert "PMID 1533045" in frame.loc[0, "evidence_found"]
    assert frame.loc[0, "pert_id"] == "BRD-A75172220"
    assert all(frame.decision.str.startswith("Insufficient"))
    assert "pubmed.ncbi.nlm.nih.gov/1533045/" in dd.pmid_links_md(frame.loc[0, "candidate_pmids"])
    assert isinstance(config["scores_available"], bool)


def test_replay_renders_embedded_trace_of_historical_run(tmp_path):
    report = {"run_id": "x", "status": "completed", "mode": "benchmark_strict",
              "tool_results": [{"tool": "rank_transcriptome", "status": "completed",
                                "manifest": "missing\\manifest.json"}],
              "trace": [{"time": "t0", "stage": "request_received"},
                        {"time": "t1", "stage": "tool_started", "tool": "rank_transcriptome"},
                        {"time": "t2", "stage": "run_completed", "tool_count": 1}]}
    path = tmp_path / "agent_run.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = dd.load_agent_run(path)
    events, integrity, trace_path = dd.load_trace_events(loaded)
    assert integrity == "embedded_only" and trace_path is None
    rows = dd.timeline_rows(events)
    assert [r["label"] for r in rows] == ["收到自然语言请求", "工具开始执行", "运行完成"]
    assert rows[1]["tool"] == "rank_transcriptome"
    refs = dd.output_references(loaded, root=tmp_path)
    assert refs[0]["存在"] is False and refs[-1]["存在"] is True


def test_saved_deepseek_run_replays_when_present():
    path = ROOT / "artifacts" / "deepseek_transcript_live" / "agent_run.json"
    if not path.is_file():
        pytest.skip("ignored artifact not present in this checkout")
    assert path.resolve() in dd.find_saved_runs()
    report = dd.load_agent_run(path)
    rows = dd.timeline_rows(dd.load_trace_events(report)[0])
    assert rows[0]["stage"] == "request_received" and rows[-1]["stage"] == "run_completed"


def test_live_rule_run_stops_safely_and_writes_traces(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-demo-test-secret-value")
    report = dd.run_demo_agent("请为肺腺癌筛选候选药物", "benchmark_strict", "rule",
                               output_root=tmp_path)
    assert report["status"] == "needs_review"
    assert report["plan"]["calls"][0]["name"] == "manual_review"
    events, integrity, trace_path = dd.load_trace_events(report)
    assert integrity == "sha256_chain_v1"
    assert [r["stage"] for r in dd.timeline_rows(events)][-1] == "manual_review_required"
    demo_trace = Path(report["_demo_trace"])
    demo_events, _ = verify_trace_chain(demo_trace, require_chain=True)
    assert demo_events[-1]["stage"] == "demo_completed"
    for trace in (demo_trace, Path(trace_path)):
        assert "sk-demo-test-secret-value" not in trace.read_text(encoding="utf-8")
    replay = dd.load_agent_run(Path(report["_path"]))
    assert replay["_demo_trace"] == str(demo_trace)


def test_deepseek_requires_environment_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert dd.deepseek_available() is False
    with pytest.raises(ValueError):
        dd.run_demo_agent("请根据转录组筛选候选药物", "benchmark_strict", "deepseek",
                          output_root=tmp_path)


def test_limitations_loaded():
    assert len(dd.load_limitations()) >= 5


def test_v2_archived_runs_render_as_plan_and_step_rows():
    runs = dd.saved_v2_runs()
    fetch = next(v for k, v in runs.items() if k.endswith("fetch_demo_deepseek"))
    plan = dd.v2_plan_rows(fetch)
    assert len(plan) == 1 and plan[0]["计划"].startswith("下载 GEO 数据并核验哈希")
    assert [r["工具"] for r in dd.v2_step_rows(fetch)][0] == "下载 GEO 数据并核验哈希"
    stopped = next(v for k, v in runs.items() if k.endswith("unregistered_demo_deepseek"))
    assert stopped["status"] == "manual_review_required"
    rows = dd.registry_rows()
    assert rows[0]["GEO"] == "GSE32863" and len(rows) == 5 and rows[3]["设计"].startswith("不配对")


def test_v2_rule_run_for_unregistered_disease_stops_without_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(dd.ROOT)
    report = dd.run_v2("请为胃癌筛选候选药物", "rule", output_root=tmp_path)
    assert report["status"] == "manual_review_required" and report["_disease"] is None
    assert [e["tool"] for e in report["executed"]] == ["manual_review"]


def test_v2_disease_label_is_recovered_for_archived_runs():
    runs = dd.saved_v2_runs()
    fetch = next(v for k, v in runs.items() if k.endswith("fetch_demo_rule"))
    stopped = next(v for k, v in runs.items() if k.endswith("unregistered_demo_deepseek"))
    assert dd.v2_disease(fetch) == "肺腺癌（LUAD）" and dd.v2_disease(stopped) is None


def test_v2_pipeline_marks_each_step_and_unrun_steps():
    report = {"rounds": [{"round": 1, "steps": [{"tool": "qc_disease_cohort"}, {"tool": "rank_candidates"},
                                                 {"tool": "build_report"}]},
                         {"round": 2, "steps": [{"tool": "rank_candidates"}], "invalid_reason": "x"},
                         {"round": 3, "steps": [{"tool": "rank_candidates"}, {"tool": "build_report"}]}],
              "executed": [{"tool": "qc_disease_cohort", "status": "ok"}, {"tool": "rank_candidates", "status": "failed"},
                           {"tool": "rank_candidates", "status": "ok"}, {"tool": "build_report", "status": "ok"}]}
    rows = dd.v2_pipeline(report)
    assert [s["status"] for s in rows[0]["steps"]] == ["ok", "failed", "not_run"]
    assert [s["status"] for s in rows[1]["steps"]] == ["not_run"] and rows[1]["invalid_reason"] == "x"
    assert [s["status"] for s in rows[2]["steps"]] == ["ok", "ok"]


def test_v2_outputs_and_header_cards_read_committed_results():
    runs = dd.saved_v2_runs()
    out = dd.v2_outputs(next(v for k, v in runs.items() if k.endswith("fetch_demo_rule")))
    assert out["signature"]["up"] == 512 and len(out["ranking"]["top10"]) == 10 and out["cohort"]["pairs"] == 57
    labels = [label for _, label, _ in dd.headline_kpis()]
    assert labels[:2] == ["登记疾病", "工具"] and "冻结多步规划" in labels
    cards = dd.registry_cards()
    assert len(cards) == 5 and all(c["reference_drugs"] > 0 for c in cards)
