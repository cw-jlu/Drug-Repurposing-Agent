"""Pair official B3/B3noREV runs with B2 and the 11 published TRANSCRIPT models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.stats import wilcoxon

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run
from scripts.compare_recess_official_b2 import (
    METRICS, MODELS, PUBLISHED_COMMIT, load_run, paired_summary, summarize,
)

OURS = {"B3": "benchmark/results/recess_official_b3",
        "B3noREV": "artifacts/recess_official_b3/results_B3noREV",
        "B2": "benchmark/results/recess_official_b2"}


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/results/recess_official_b3_vs_11.json"))
    args = parser.parse_args()
    published = args.published.resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(published), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != PUBLISHED_COMMIT:
        raise ValueError(f"Expected published results commit {PUBLISHED_COMMIT}; got {head}")
    repo = Path(__file__).resolve().parents[1]
    seeds = np.random.RandomState(1234).choice(range(int(1e8)), size=100).tolist()
    report = {
        "protocol": "RECeSS TRANSCRIPT official runner, N=100, K=5, ptest=0.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "Lin's AUC (paper NS-AUC)",
        "b3_config": "configs/b3_row_fusion_v1.json",
        "b3_config_sha256": sha256_file(repo / "configs" / "b3_row_fusion_v1.json"),
        "b3_implementation_sha256": sha256_file(
            repo / "benchmarks" / "recess_adapter" / "official_b3.py"),
        "b3_patch_sha256": sha256_file(
            repo / "benchmarks" / "recess_adapter" / "official_b3.patch"),
        "published_results_commit": head,
        "notes": ["B3 is the primary model frozen before official scoring.",
                  "B3noREV is post-hoc: its weak-split development score was seen first.",
                  "weakly_correlated repeats one outer holdout across all 100 seeds; its "
                  "seed-paired tests are not independent replications."],
        "splits": {}, "trace_file": str(trace.path),
    }
    for split in ("random_simple", "weakly_correlated"):
        reference_dir = published / ("results_TRANSCRIPT" if split == "random_simple"
                                     else "results_TRANSCRIPT_weakly_correlated")
        runs = {}
        for name, folder in OURS.items():
            if not any((repo / folder).glob(f"results_N=100_{name}_TRANSCRIPT_{split}_*.csv")):
                trace.emit("model_skipped_incomplete", model=name, split=split)
                print(f"  skip {name}: no completed N=100 {split} result")
                continue
            runs[name] = load_run((repo / folder).resolve(strict=True), name, split, seeds)
        for model in MODELS:
            runs[model] = load_run(reference_dir / f"results_{model}", model, split, seeds)
        rows = {name: {m: summarize(v) for m, v in run["series"].items()}
                for name, run in runs.items()}
        ranking = sorted(rows, key=lambda n: -(rows[n]["NS-AUC"]["mean"] or -1))
        best_published = next(n for n in ranking if n in MODELS)
        paired = {}
        for ours in [n for n in ("B3", "B3noREV") if n in runs]:
            paired[ours] = {}
            for other in [best_published, "B2"]:
                a = runs[ours]["series"]["NS-AUC"]; b = runs[other]["series"]["NS-AUC"]
                ok = np.isfinite(a) & np.isfinite(b); d = a[ok] - b[ok]
                stat = {"wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                        "ties": int((d == 0).sum()), **paired_summary(d)}
                if np.any(d != 0):
                    stat["wilcoxon_p"] = float(wilcoxon(d[d != 0]).pvalue)
                paired[ours][other] = stat
        report["splits"][split] = {
            "ranking_by_ns_auc": [{"model": n, "rank": i + 1, "ns_auc_mean": rows[n]["NS-AUC"]["mean"],
                                   "ns_auc_sd": rows[n]["NS-AUC"].get("sd")}
                                  for i, n in enumerate(ranking)],
            "best_published": best_published, "models": rows, "paired_ns_auc": paired,
            "source_sha256": {n: r["source_files"] for n, r in runs.items()},
        }
        trace.emit("split_compared", split=split, ranking=ranking)
        print(split)
        for i, n in enumerate(ranking):
            print(f"  {i+1:2d}. {n:24s} {rows[n]['NS-AUC']['mean']:.4f}")
        print("  paired:", json.dumps(paired, default=float)[:600])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    trace.emit("report_saved", output=str(args.output), sha256=sha256_file(args.output))


def main() -> None:
    traced_run("recess_b3_comparison", _main)


if __name__ == "__main__":
    main()
