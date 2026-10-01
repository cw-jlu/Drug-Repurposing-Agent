"""Secondary metrics and overlap-aware paired statistics for the official TRANSCRIPT runs.

Reads only committed/pinned per-seed CSVs: our B2/B3/B3noREV/B4 runs and the 11
published RECeSS models. No model is re-run and no parameter is changed.

The 100 random_simple seeds draw overlapping 20% outer test sets from one dataset,
so seed-level differences are not independent. Besides the descriptive paired
summary and the naive Wilcoxon test, we report the Nadeau-Bengio corrected
resampled t-test (variance inflated by 1/J + n_test/n_train). The weakly
correlated split reuses a single outer holdout for every seed, so no
generalisation test is meaningful there; only descriptive values are reported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run

PUBLISHED = ("ALSWR", "BNNR", "DDA_SKF", "FastaiCollabWrapper", "HAN", "LibMF",
             "LogisticMF", "MBiRW", "NIMCGCN", "PMF", "SCPMF")
OURS = {"B2": "benchmark/results/recess_official_b2",
        "B3": "benchmark/results/recess_official_b3",
        "B3noREV": "benchmark/results/recess_official_b3",
        "B4": "benchmark/results/recess_official_b4"}
METRICS = {"NS-AUC": "Lin's AUC", "global AUC": "global AUC", "global NDCG": "global NDCG",
           "HR@10": "HR@10", "training time (s)": "training time (sec)"}
SPLITS = ("random_simple", "weakly_correlated")
PTEST = 0.2


def _csv(folder: Path, model: str, split: str) -> Path:
    return folder / f"results_N=100_{model}_TRANSCRIPT_{split}_AUC_1.000000_5_0.200000.csv"


def load(folder: Path, model: str, split: str) -> tuple[pd.DataFrame, Path]:
    path = _csv(folder, model, split)
    frame = pd.read_csv(path, index_col=0)
    if frame.shape[1] != 100:
        raise ValueError(f"{path} has {frame.shape[1]} seeds, expected 100")
    return frame, path


def nadeau_bengio(diff: np.ndarray, ptest: float = PTEST) -> dict:
    j = len(diff)
    mean, var = float(diff.mean()), float(diff.var(ddof=1))
    corrected = (1 / j + ptest / (1 - ptest)) * var
    t = mean / np.sqrt(corrected) if corrected > 0 else float("inf")
    p = float(2 * stats.t.sf(abs(t), df=j - 1))
    half = float(stats.t.ppf(0.975, df=j - 1) * np.sqrt(corrected))
    return {"t": float(t), "p_two_sided": p, "mean_ci95": [mean - half, mean + half]}


def paired(a: np.ndarray, b: np.ndarray, split: str) -> dict:
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    out = {"n": int(len(d)), "mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
           "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}
    if split == "random_simple":
        nz = d[d != 0]
        out["naive_wilcoxon_p"] = float(stats.wilcoxon(nz).pvalue) if len(nz) else None
        out["nadeau_bengio"] = nadeau_bengio(d)
    else:
        out["note"] = "single repeated outer holdout: descriptive only"
    return out


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published", type=Path,
                        default=Path(os.environ.get("TEMP", "/tmp")) / "recess-benchmark-results-audit-20260925")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/benchmark_stats_v1.json"))
    args = parser.parse_args()
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "protocol": "RECeSS official runner per-seed CSVs, N=100, K=5, ptest=0.2",
              "nadeau_bengio_correction": "var * (1/J + n_test/n_train), J=100, n_test/n_train=0.25",
              "splits": {}, "sources": {}, "trace_file": str(trace.path)}
    for split in SPLITS:
        ref = args.published / ("results_TRANSCRIPT" if split == "random_simple"
                                else "results_TRANSCRIPT_weakly_correlated")
        frames = {}
        for model in PUBLISHED:
            frames[model], path = load(ref / f"results_{model}", model, split)
            report["sources"][f"{split}/{model}"] = sha256_file(path)
        for model, folder in OURS.items():
            frames[model], path = load(Path(folder), model, split)
            report["sources"][f"{split}/{model}"] = sha256_file(path)
        table = {}
        for model, frame in frames.items():
            row = {}
            for label, key in METRICS.items():
                values = frame.loc[key].to_numpy(dtype=float)
                row[label] = {"mean": float(np.nanmean(values)), "sd": float(np.nanstd(values, ddof=1))}
            table[model] = row
        ns = {m: frames[m].loc["Lin's AUC"].to_numpy(dtype=float) for m in frames}
        best = max(PUBLISHED, key=lambda m: np.nanmean(ns[m]))
        comparisons = {}
        for ours in ("B4", "B3"):
            for other in sorted({best, "BNNR", "MBiRW", "B2"} - {ours}):
                comparisons[f"{ours} - {other}"] = paired(ns[ours], ns[other], split)
        report["splits"][split] = {"best_published_ns_auc": best, "metrics": table,
                                   "paired_ns_auc": comparisons}
        trace.emit("split_done", split=split, best=best)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))
    for split, body in report["splits"].items():
        print(split, "best published:", body["best_published_ns_auc"])
        for k, v in body["paired_ns_auc"].items():
            nb = v.get("nadeau_bengio")
            print(f"  {k:16s} diff={v['mean_diff']:+.4f} wins={v['wins']}/{v['n']}"
                  + (f" naiveW p={v['naive_wilcoxon_p']:.2e} NB p={nb['p_two_sided']:.3f} CI=[{nb['mean_ci95'][0]:+.4f},{nb['mean_ci95'][1]:+.4f}]" if nb else ""))


def main() -> None:
    traced_run("benchmark_stats_v1", _main)


if __name__ == "__main__":
    main()
