from evals.agent_repeatability import summarize


def test_summary_gates():
    runs = [{"status": "completed", "plan_calls": ["rank_transcriptome"], "rrf_sha256": "a",
             "spearman_vs_first": 1.0, "seconds": 1.0} for _ in range(9)]
    runs.append({"status": "needs_review", "plan_calls": ["manual_review"], "seconds": 0.0})
    s = summarize(runs, None)
    assert s["completion_rate"] == 0.9 and s["gate_completion_ge_0.90"]
    assert s["gate_spearman_ge_0.95"] and s["distinct_rrf_hashes"] == 1
