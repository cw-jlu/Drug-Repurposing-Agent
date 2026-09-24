"""Run fixed baselines with RECeSS's split and validation functions.

Usage: python benchmarks/recess_adapter/run.py --data data/raw/TRANSCRIPT_dataset_v2.0.0
This script owns labels. The drug_repurposing_agent workflow does not import it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import stanscofi.datasets
import stanscofi.training_testing
import stanscofi.validation
from sklearn.metrics import roc_auc_score

from drug_repurposing_agent.benchmark import TranscriptBaseline
from drug_repurposing_agent.data import sha256_file


def run(data_dir: Path, split: str, seed: int) -> dict:
    ratings_path = data_dir / "ratings_mat.csv"
    items_path = data_dir / "items.csv"
    users_path = data_dir / "users.csv"
    ratings = pd.read_csv(ratings_path, index_col=0)
    items = pd.read_csv(items_path, index_col=0)
    users = pd.read_csv(users_path, index_col=0)
    dataset = stanscofi.datasets.Dataset(ratings=ratings, items=items, users=users,
                                         name="TRANSCRIPT")
    splitter = getattr(stanscofi.training_testing, split + "_split")
    (train_folds, test_folds), _ = splitter(dataset, 0.2, metric="euclidean", random_state=seed)
    if (train_folds.multiply(test_folds)).nnz:
        raise RuntimeError("Training and validation folds overlap")
    train = dataset.subset(train_folds)
    test = dataset.subset(test_folds)
    truth = test.ratings.toarray()[test.folds.row, test.folds.col]
    binary_truth = (truth == 1).astype(int)
    if len(np.unique(binary_truth)) < 2:
        raise RuntimeError("Validation fold has only one scoring class")
    result = {"dataset": "TRANSCRIPT v2.0.0", "split": split,
              "seed": seed, "test_size": 0.2,
              "counts": {"train_pairs": int(train_folds.nnz), "test_pairs": int(test_folds.nnz),
                         "test_positive": int((truth == 1).sum()),
                         "test_explicit_negative": int((truth == -1).sum()),
                         "test_unknown": int((truth == 0).sum())},
              "input_sha256": {"ratings": sha256_file(ratings_path),
                               "items": sha256_file(items_path),
                               "users": sha256_file(users_path)},
              "methods": {}}
    for method in ("B0", "B0p", "B1", "B1k", "B2"):
        start = perf_counter()
        model = TranscriptBaseline({"method": method, "seed": seed}).fit(train)
        scores = model.predict_proba(test)
        if not np.array_equal(scores.row, test.folds.row) or not np.array_equal(scores.col, test.folds.col):
            raise RuntimeError("Score coordinates differ from official test fold")
        global_auc = stanscofi.validation.AUC(binary_truth, scores.data, 1, 1)
        global_ndcg = stanscofi.validation.NDCGk(binary_truth, scores.data, len(scores.data), 1)
        score_matrix = scores.toarray()
        rating_matrix = test.ratings.toarray()
        fold_matrix = test.folds.toarray().astype(bool)
        row_auc = []
        recall10 = []
        reciprocal_rank = []
        for disease_index in range(score_matrix.shape[1]):
            mask = fold_matrix[:, disease_index]
            labels = (rating_matrix[mask, disease_index] == 1).astype(int)
            if not labels.any():
                continue
            ranks = np.argsort(-score_matrix[mask, disease_index], kind="stable")
            recall10.append(float(labels[ranks[:10]].sum() / labels.sum()))
            reciprocal_rank.append(float(1 / (np.flatnonzero(labels[ranks])[0] + 1)))
            if len(np.unique(labels)) == 2:
                row_auc.append(float(roc_auc_score(labels, score_matrix[mask, disease_index])))
        result["methods"][method] = {"global_AUC": float(global_auc),
                                      "global_NDCG": float(global_ndcg),
                                      "rowwise_score_AUC_mean": float(np.mean(row_auc)) if row_auc else None,
                                      "rowwise_AUC_diseases": len(row_auc),
                                      "recall_at_10_mean": float(np.mean(recall10)),
                                      "MRR_mean": float(np.mean(reciprocal_rank)),
                                      "runtime_sec": round(perf_counter() - start, 3)}
        print(f"{split} seed={seed} {method}: AUC={global_auc:.4f} NDCG={global_ndcg:.4f}",
              flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", choices=["random_simple", "weakly_correlated"],
                        default="random_simple")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark_results"))
    args = parser.parse_args()
    result = run(args.data, args.split, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"transcript_{args.split}_seed{args.seed}.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
