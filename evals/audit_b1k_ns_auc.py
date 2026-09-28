"""Trace a single frozen official B1k seed down to positive/negative score pairs.

This is a diagnostic on an already reported split, never a model-selection run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from benchscofi.utils import rowwise_metrics
from stanscofi.datasets import Dataset
from stanscofi.training_testing import cv_training, random_simple_split, weakly_correlated_split

from benchmarks.recess_adapter.official_components import B1k
from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


def pair_counts(scores, truth, folds) -> dict[str, int]:
    """Replicate Lin's strict pairwise comparison and expose score ties."""
    coords = folds.tocoo()
    values = scores.toarray()[coords.row, coords.col]
    labels = truth.toarray()[coords.row, coords.col]
    counts = {"wins": 0, "ties": 0, "losses": 0, "eligible_rows": 0,
              "ignored_rows": 0}
    row_half_tie_aucs = []
    row_inverted_strict_aucs = []
    for row in np.unique(coords.row):
        take = coords.row == row
        positive = values[take & (labels == 1)]
        negative = values[take & (labels < 1)]
        if not len(positive) or not len(negative) or len(np.unique(labels[take])) != 2:
            counts["ignored_rows"] += 1
            continue
        counts["eligible_rows"] += 1
        difference = positive[:, None] - negative[None, :]
        counts["wins"] += int(np.count_nonzero(difference > 0))
        counts["ties"] += int(np.count_nonzero(difference == 0))
        counts["losses"] += int(np.count_nonzero(difference < 0))
        row_half_tie_aucs.append(float(np.mean(difference > 0) +
                                        np.mean(difference == 0) / 2))
        row_inverted_strict_aucs.append(float(np.mean(difference < 0)))
    counts["row_mean_half_tie_auc"] = float(np.mean(row_half_tie_aucs))
    counts["row_mean_inverted_strict_auc"] = float(np.mean(row_inverted_strict_aucs))
    return counts


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("random_simple", "weakly_correlated"),
                        default="random_simple")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--data", type=Path,
                        default=Path("data/raw/TRANSCRIPT_dataset_v2.0.0"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/reports/b1k_ns_auc_diagnostic.json"))
    args = parser.parse_args()
    if not 0 <= args.seed_index < 100:
        raise ValueError("seed-index must be 0..99")
    root = Path("benchmark/results/recess_official_components/results_B1k")
    stem = f"N=100_B1k_TRANSCRIPT_{args.split}_AUC_1.000000_5_0.200000.csv"
    seeds_path = root / f"seeds_{stem}"
    results_path = root / f"results_{stem}"
    seeds = pd.read_csv(seeds_path, index_col=0).loc["seed"].astype(int).tolist()
    seed = seeds[args.seed_index]
    trace.emit("frozen_inputs", split=args.split, seed_index=args.seed_index, seed=seed,
               seeds_sha256=sha256_file(seeds_path), results_sha256=sha256_file(results_path),
               data_sha256={name: sha256_file(args.data / name)
                            for name in ("ratings_mat.csv", "items.csv", "users.csv")})
    ratings = pd.read_csv(args.data / "ratings_mat.csv", index_col=0).fillna(0).astype(int)
    ratings.index = ratings.index.astype(str)
    users = pd.read_csv(args.data / "users.csv", index_col=0).astype(float)
    items = pd.read_csv(args.data / "items.csv", index_col=0).astype(float)
    users.columns = users.columns.astype(str)
    items.columns = items.columns.astype(str)
    dataset = Dataset(ratings=ratings, users=users, items=items, name="TRANSCRIPT")
    split_fn = random_simple_split if args.split == "random_simple" else weakly_correlated_split
    (train_folds, validation_folds), _ = split_fn(
        dataset, 0.2, metric="euclidean", random_state=seed)
    train = dataset.subset(train_folds)
    selected = cv_training(B1k, None, train, 5, "AUC", k=1, beta=1,
                           threshold=0, cv_type="random", random_state=seed)
    model = selected["models"][np.argmax(selected["test_metric"])]
    validation = dataset.subset(validation_folds)
    scores = model.predict_proba(validation)
    official_rows = rowwise_metrics.calc_auc(scores, validation, transpose=False)
    official = float(np.mean(official_rows)) if max(official_rows, default=0) > 0 else 0.5
    saved = float(pd.read_csv(results_path, index_col=0)
                  .loc["Lin's AUC"].iloc[args.seed_index])
    if not np.isclose(official, saved, rtol=0, atol=1e-12):
        raise RuntimeError(f"Reproduction differs from saved official result: {official}, {saved}")
    pairs = pair_counts(scores, validation.ratings, validation.folds)
    total = pairs["wins"] + pairs["ties"] + pairs["losses"]
    diagnostic = {"split": args.split, "seed_index": args.seed_index,
                  "seed": seed, "saved_official_ns_auc": saved,
                  "reproduced_official_ns_auc": official,
                  "pair_counts": pairs, "pairwise_strict_win_fraction": pairs["wins"] / total,
                  "pairwise_half_tie_fraction": (pairs["wins"] + pairs["ties"] / 2) / total,
                  "pairwise_inverted_strict_fraction": pairs["losses"] / total,
                  "interpretation": "Diagnostic only: official NS-AUC is an unweighted mean of eligible drug-row strict pairwise AUCs; ties receive zero. Pair fractions pool pairs and are not NS-AUC. Half-tie and inverted figures are counterfactual diagnostics, not replacement official results."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    trace.emit("diagnostic_completed", output=str(args.output),
               output_sha256=sha256_file(args.output), **diagnostic)
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    traced_run("b1k_ns_auc_diagnostic", _main)
