"""Reproduce frozen official B1k seeds and measure strict-score ties.

Retrospective diagnostic only. The official comparison and its metric are not changed.
Each seed is checkpointed and emitted to a SHA-256 chained trace immediately.
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
from evals.audit_b1k_ns_auc import pair_counts


def load_dataset(data_dir: Path) -> Dataset:
    ratings = pd.read_csv(data_dir / "ratings_mat.csv", index_col=0).fillna(0).astype(int)
    ratings.index = ratings.index.astype(str)
    users = pd.read_csv(data_dir / "users.csv", index_col=0).astype(float)
    items = pd.read_csv(data_dir / "items.csv", index_col=0).astype(float)
    users.columns = users.columns.astype(str)
    items.columns = items.columns.astype(str)
    return Dataset(ratings=ratings, users=users, items=items, name="TRANSCRIPT")


def audit_seed(dataset: Dataset, split: str, seed: int, saved: float) -> dict:
    split_fn = random_simple_split if split == "random_simple" else weakly_correlated_split
    (train_folds, validation_folds), _ = split_fn(
        dataset, 0.2, metric="euclidean", random_state=seed)
    train = dataset.subset(train_folds)
    selected = cv_training(B1k, None, train, 5, "AUC", k=1, beta=1,
                           threshold=0, cv_type="random", random_state=seed)
    model = selected["models"][np.argmax(selected["test_metric"])]
    validation = dataset.subset(validation_folds)
    scores = model.predict_proba(validation)
    row_aucs = rowwise_metrics.calc_auc(scores, validation, transpose=False)
    official = float(np.mean(row_aucs)) if max(row_aucs, default=0) > 0 else 0.5
    if not np.isclose(official, saved, rtol=0, atol=1e-12):
        raise RuntimeError(f"Official score differs: seed={seed}, reproduced={official}, saved={saved}")
    pairs = pair_counts(scores, validation.ratings, validation.folds)
    total = pairs["wins"] + pairs["ties"] + pairs["losses"]
    if total == 0:
        raise RuntimeError(f"No eligible pairs for seed {seed}")
    return {"seed": seed, "official_ns_auc": official,
            "eligible_drug_rows": pairs["eligible_rows"],
            "ignored_drug_rows_with_test_coordinates": pairs["ignored_rows"],
            "pair_wins": pairs["wins"], "pair_ties": pairs["ties"],
            "pair_losses": pairs["losses"], "tie_fraction": pairs["ties"] / total,
            "row_mean_half_tie_auc": pairs["row_mean_half_tie_auc"],
            "row_mean_inverted_strict_auc": pairs["row_mean_inverted_strict_auc"]}


def summarize(rows: list[dict]) -> dict:
    return {"seed_count": len(rows),
            "official_ns_auc_mean": float(np.mean([r["official_ns_auc"] for r in rows])),
            "tie_fraction_mean": float(np.mean([r["tie_fraction"] for r in rows])),
            "tie_fraction_median": float(np.median([r["tie_fraction"] for r in rows])),
            "tie_fraction_min": float(np.min([r["tie_fraction"] for r in rows])),
            "tie_fraction_max": float(np.max([r["tie_fraction"] for r in rows])),
            "half_tie_row_auc_mean": float(np.mean([r["row_mean_half_tie_auc"] for r in rows])),
            "inverted_strict_row_auc_mean": float(np.mean([r["row_mean_inverted_strict_auc"] for r in rows])),
            "eligible_drug_rows_mean": float(np.mean([r["eligible_drug_rows"] for r in rows]))}


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=("random_simple", "weakly_correlated"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--data", type=Path,
                        default=Path("data/raw/TRANSCRIPT_dataset_v2.0.0"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.start + args.count > 100:
        raise ValueError("Requested seed indices must be within 0..99")
    checkpoint = args.output.with_suffix(".jsonl")
    if args.output.exists() or checkpoint.exists():
        raise ValueError("Output or seed checkpoint already exists; refusing overwrite")
    root = Path("benchmark/results/recess_official_components/results_B1k")
    stem = f"N=100_B1k_TRANSCRIPT_{args.split}_AUC_1.000000_5_0.200000.csv"
    seed_file, result_file = root / f"seeds_{stem}", root / f"results_{stem}"
    seeds = pd.read_csv(seed_file, index_col=0).loc["seed"].astype(int).tolist()
    saved = pd.read_csv(result_file, index_col=0).loc["Lin's AUC"].astype(float).tolist()
    if len(seeds) != len(saved) or len(seeds) != 100:
        raise ValueError("Frozen official seed/results columns do not match 100 runs")
    trace.emit("frozen_inputs", split=args.split, start=args.start, count=args.count,
               seed_file_sha256=sha256_file(seed_file), result_file_sha256=sha256_file(result_file),
               data_sha256={name: sha256_file(args.data / name)
                            for name in ("ratings_mat.csv", "items.csv", "users.csv")})
    dataset = load_dataset(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(args.start, args.start + args.count):
        row = {"seed_index": index, **audit_seed(dataset, args.split, seeds[index], saved[index])}
        with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        trace.emit("seed_reproduced", **row)
        rows.append(row)
        if len(rows) % 10 == 0:
            print(f"{args.split}: {len(rows)}/{args.count} frozen seeds reproduced", flush=True)
    report = {"status": "retrospective_diagnostic_not_new_benchmark",
              "split": args.split, "start_index": args.start, "count": args.count,
              "summary": summarize(rows), "seeds": rows,
              "checkpoint_file": str(checkpoint),
              "checkpoint_sha256": sha256_file(checkpoint),
              "trace_file": str(trace.path)}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("audit_saved", output=str(args.output), output_sha256=sha256_file(args.output),
               checkpoint_sha256=report["checkpoint_sha256"], summary=report["summary"])
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    traced_run("b1k_all_seeds_audit", _main)
