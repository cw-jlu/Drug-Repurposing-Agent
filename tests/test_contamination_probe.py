"""Deterministic parts of the contamination probe: sampling freeze, blinding, AUC, resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drug_repurposing_agent.llm_calls import CallStats, TracedToolCaller
from evals.contamination_probe import (analyze, assert_closed_book_blind, auc, bootstrap_auc_ci,
                                       build_sample, closed_book_payload, freeze_sample,
                                       load_checkpoint, load_frozen_sample, open_book_payload,
                                       run_calls, validate_probability)
from drug_repurposing_agent.trace import TraceRecorder


def toy_inputs():
    drugs = [f"DB{i:05d}" for i in range(12)]
    diseases = [f"C{j:07d}" for j in range(8)]
    values = np.zeros((12, 8), dtype=int)
    for i in range(12):
        values[i, i % 8] = 1
    values[0, 1] = -1
    values[3, 5] = -1
    ratings = pd.DataFrame(values, index=drugs, columns=diseases)
    drug_names = {d: {"name": f"Drugname{i}", "source": "t"} for i, d in enumerate(drugs) if i != 11}
    disease_names = {c: {"name": f"Diseasename{j}", "source": "t"} for j, c in enumerate(diseases)}
    rng = np.random.default_rng(0)
    score = pd.DataFrame(rng.normal(size=(12, 8)), index=drugs, columns=diseases)
    pct = score.rank(axis=1).sub(1).div(7)
    return ratings, drug_names, disease_names, score, pct


def fake_transport(probability=0.7):
    calls = []

    def transport(payload):
        calls.append(payload)
        return {"model": "deepseek-flash", "usage": {"prompt_tokens": 10, "completion_tokens": 2,
                                                     "total_tokens": 12},
                "choices": [{"finish_reason": "tool_calls", "message": {"tool_calls": [
                    {"function": {"name": "submit_probability",
                                  "arguments": json.dumps({"probability": probability})}}]}}]}
    return transport, calls


def test_auc_matches_mann_whitney_and_handles_ties():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(0.3, 1, 50).round(1), rng.normal(0, 1, 60).round(1)
    expected = sklearn.roc_auc_score([1] * 50 + [0] * 60, np.r_[pos, neg])
    assert auc(pos, neg) == pytest.approx(expected)


def test_bootstrap_ci_is_seeded_and_brackets_point_estimate():
    rng = np.random.default_rng(2)
    pos, neg = rng.normal(1, 1, 40), rng.normal(0, 1, 40)
    first, second = bootstrap_auc_ci(pos, neg, n_boot=300), bootstrap_auc_ci(pos, neg, n_boot=300)
    assert first == second
    assert first["ci95"][0] <= first["auc"] <= first["ci95"][1]


def test_sample_is_deterministic_capped_balanced_and_excludes_unmapped():
    ratings, drugs, diseases, score, pct = toy_inputs()
    rows = build_sample(ratings, drugs, diseases, score, pct, seed=7, positive_cap=5)
    assert rows == build_sample(ratings, drugs, diseases, score, pct, seed=7, positive_cap=5)
    labels = [r["label"] for r in rows]
    assert labels.count(1) == 5 and labels.count(0) == 5 and labels.count(-1) == 2
    assert all(r["drug_id"] != "DB00011" for r in rows)
    assert len({(r["drug_id"], r["disease_id"]) for r in rows}) == len(rows)
    assert all(0 <= r["reversal_drug_percentile"] <= 1 for r in rows)


def test_freeze_refuses_resampling_and_detects_tampering(tmp_path: Path):
    ratings, drugs, diseases, score, pct = toy_inputs()
    rows = build_sample(ratings, drugs, diseases, score, pct, seed=7, positive_cap=5)
    path = tmp_path / "sample.jsonl"
    digest = freeze_sample(rows, path)
    assert freeze_sample(rows, path) == digest
    loaded, loaded_digest = load_frozen_sample(path)
    assert loaded == rows and loaded_digest == digest
    other = build_sample(ratings, drugs, diseases, score, pct, seed=8, positive_cap=5)
    with pytest.raises(ValueError, match="refusing to resample"):
        freeze_sample(other, path)
    path.write_bytes(path.read_bytes().replace(b"Drugname", b"Xrugname", 1))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_sample(path)


def test_closed_book_payload_is_blind_and_open_book_uses_names():
    ratings, drugs, diseases, score, pct = toy_inputs()
    row = build_sample(ratings, drugs, diseases, score, pct, seed=7, positive_cap=5)[0]
    closed = closed_book_payload(row, "deepseek-flash")
    assert_closed_book_blind(closed, row)
    text = json.dumps(closed)
    assert row["drug_name"] not in text and row["drug_id"] not in text and '"label"' not in text
    assert str(row["reversal_score"]) in text
    assert row["drug_name"] in json.dumps(open_book_payload(row, "deepseek-flash"))
    leaky = json.loads(json.dumps(closed))
    leaky["messages"][1]["content"] += " " + row["disease_name"]
    with pytest.raises(ValueError, match="leaks"):
        assert_closed_book_blind(leaky, row)


@pytest.mark.parametrize("bad", [{"probability": 1.5}, {"probability": -0.1},
                                 {"probability": True}, {"probability": "0.3"}, {}])
def test_probability_validation_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        validate_probability(bad)


def test_run_resumes_from_checkpoint_and_analyze_aggregates(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DRUG_AGENT_TRACE_DIR", str(tmp_path / "provider"))
    ratings, drugs, diseases, score, pct = toy_inputs()
    rows = build_sample(ratings, drugs, diseases, score, pct, seed=7, positive_cap=5)
    digest = freeze_sample(rows, tmp_path / "sample.jsonl")
    transport, calls = fake_transport()
    trace = TraceRecorder("probe_test", tmp_path / "traces")
    caller = TracedToolCaller(None, transport=transport, backoff_seconds=0, stats=CallStats())
    checkpoint = tmp_path / "checkpoint.jsonl"
    stats = run_calls(trace, caller, rows, digest, checkpoint, "deepseek-flash", workers=2)
    assert stats["attempts"] == 2 * len(rows) == len(calls)
    closed = [c for c in calls if "features" in c["messages"][1]["content"]]
    assert all("Drugname" not in c["messages"][1]["content"] for c in closed)
    again = run_calls(trace, TracedToolCaller(None, transport=transport, backoff_seconds=0),
                      rows, digest, checkpoint, "deepseek-flash")
    assert again["attempts"] == 0 and len(calls) == 2 * len(rows)
    records = load_checkpoint(checkpoint, digest)
    with pytest.raises(ValueError, match="different frozen sample"):
        load_checkpoint(checkpoint, "0" * 64)
    result = analyze(rows, records, digest, "deepseek-flash", {}, stats, str(trace.path), None)
    assert result["conditions"]["open_book"]["positive_vs_unknown"]["auc"] == 0.5
    assert "messages" not in json.dumps(result)
    assert result["sample"]["sha256"] == digest


def test_run_refuses_to_exceed_call_budget(tmp_path: Path):
    ratings, drugs, diseases, score, pct = toy_inputs()
    rows = build_sample(ratings, drugs, diseases, score, pct, seed=7, positive_cap=5)
    transport, calls = fake_transport()
    caller = TracedToolCaller(None, transport=transport, backoff_seconds=0)
    with pytest.raises(ValueError, match="budget"):
        run_calls(TraceRecorder("probe_test", tmp_path), caller, rows, "x" * 64,
                  tmp_path / "c.jsonl", "deepseek-flash", max_calls=3)
    assert not calls
