"""Nested tuning for the three official benchscofi matrix-factorization baselines."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import chain
from io import StringIO
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.sparse import coo_array

# benchscofi 2.0.1 still references the NumPy alias removed in NumPy 2.
if not hasattr(np, "int"):
    np.int = int

from benchscofi.ALSWR import ALSWR
from benchscofi.LogisticMF import LogisticMF
from benchscofi.PMF import PMF

from benchmarks.recess_adapter.nested_cv import (
    LocalDataset,
    fold_coordinates,
    metric_values,
    random_cv_split,
    random_simple_split,
    weakly_correlated_split,
)
from drug_repurposing_agent.data import sha256_file
from sklearn.model_selection import StratifiedKFold


MODELS = {"ALSWR": ALSWR, "PMF": PMF, "LogisticMF": LogisticMF}


def make_model(name: str, candidate: dict, seed: int):
    model_class = MODELS[name]
    defaults = dict(model_class().default_parameters())
    defaults.update(candidate)
    if name == "ALSWR":
        defaults["random_state"] = seed
    return model_class(defaults)


def neutral_subset(dataset: LocalDataset, folds) -> LocalDataset:
    ratings = pd.DataFrame(0.0, index=dataset.item_list, columns=dataset.user_list)
    items = pd.DataFrame(dataset.items.toarray(), index=dataset.item_features,
                         columns=dataset.item_list)
    users = pd.DataFrame(dataset.users.toarray(), index=dataset.user_features,
                         columns=dataset.user_list)
    return LocalDataset(ratings, items, users, folds=folds,
                        name=f"{dataset.name}_label_neutral_prediction")


def predict_label_neutral(model, dataset: LocalDataset):
    """Use the official wrapper with a feature-identical, all-zero rating matrix."""
    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return model.predict_proba(dataset)


def fit_quietly(model, train, seed: int) -> None:
    """Suppress progress bars emitted by the pinned third-party algorithms."""
    sink = StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        model.fit(train, seed=seed)


def training_subset(dataset: LocalDataset, folds) -> LocalDataset:
    """Create a fold subset whose unavailable sparse ratings are numeric zeros."""
    subset = dataset.subset(folds)
    subset.ratings = coo_array(np.nan_to_num(subset.ratings.toarray(), nan=0.0))
    return subset


def evaluate_candidate(name: str, candidate: dict, outer_train: LocalDataset,
                       inner_splits, seed: int) -> dict:
    folds = []
    for index, (train_folds, validation_folds) in enumerate(inner_splits):
        train = training_subset(outer_train, train_folds)
        validation_truth = outer_train.subset(validation_folds)
        validation_features = neutral_subset(outer_train, validation_folds)
        model = make_model(name, candidate, seed + index)
        fit_quietly(model, train, seed + index)
        scores = predict_label_neutral(model, validation_features)
        values = metric_values(validation_truth, scores)
        folds.append({"fold": index, **values,
                      "train_pairs": int(train_folds.nnz),
                      "validation_pairs": int(validation_folds.nnz)})
    return {
        "parameters": candidate,
        "inner_folds": folds,
        "mean_global_AUC": float(np.mean([row["global_AUC"] for row in folds])),
        "mean_global_NDCG": float(np.mean([row["global_NDCG"] for row in folds])),
    }


def run(data_dir: Path, grid_path: Path, split: str, seed: int) -> dict:
    started = perf_counter()
    ratings_path = data_dir / "ratings_mat.csv"
    items_path = data_dir / "items.csv"
    users_path = data_dir / "users.csv"
    dataset = LocalDataset(
        ratings=pd.read_csv(ratings_path, index_col=0),
        items=pd.read_csv(items_path, index_col=0),
        users=pd.read_csv(users_path, index_col=0),
        name="TRANSCRIPT",
    )
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    split_fn = random_simple_split if split == "random_simple" else weakly_correlated_split
    outer_train_folds, outer_test_folds = split_fn(dataset, 0.2, seed)
    if outer_train_folds.multiply(outer_test_folds).nnz:
        raise RuntimeError("Outer training and test folds overlap")
    outer_train = training_subset(dataset, outer_train_folds)
    outer_test = dataset.subset(outer_test_folds)
    outer_test_features = neutral_subset(dataset, outer_test_folds)
    outer_labels = outer_train.ratings.toarray()[outer_train.folds.row,
                                                 outer_train.folds.col]
    cv = StratifiedKFold(n_splits=int(grid["inner_folds"]), shuffle=True,
                         random_state=seed + 10000)
    inner_splits = random_cv_split(outer_train, cv)
    outer_test_coords = fold_coordinates(outer_test_folds)
    inner_coords = set(chain.from_iterable(
        fold_coordinates(train) | fold_coordinates(validation)
        for train, validation in inner_splits
    ))
    if outer_test_coords & inner_coords:
        raise RuntimeError("Outer test coordinates leaked into inner CV")

    methods = {}
    for name, candidates in grid["methods"].items():
        evaluated = []
        for candidate in candidates:
            result = evaluate_candidate(name, candidate, outer_train, inner_splits, seed)
            evaluated.append(result)
            print(f"{split} seed={seed} {name} {candidate}: "
                  f"inner AUC={result['mean_global_AUC']:.4f}", flush=True)
        selected = sorted(
            evaluated,
            key=lambda row: (-row["mean_global_AUC"], -row["mean_global_NDCG"],
                             json.dumps(row["parameters"], sort_keys=True)),
        )[0]
        model = make_model(name, selected["parameters"], seed)
        fit_quietly(model, outer_train, seed)
        outer_scores = predict_label_neutral(model, outer_test_features)
        methods[name] = {
            "selected_parameters": selected["parameters"],
            "selection": {
                "metric": grid["selection_metric"],
                "mean_global_AUC": selected["mean_global_AUC"],
                "mean_global_NDCG": selected["mean_global_NDCG"],
            },
            "outer_test": metric_values(outer_test, outer_scores),
            "candidate_results": evaluated,
        }
        print(f"selected {name} {selected['parameters']}: "
              f"outer AUC={methods[name]['outer_test']['global_AUC']:.4f}", flush=True)

    return {
        "protocol": grid["name"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "TRANSCRIPT v2.0.0",
        "split": split,
        "seed": seed,
        "outer_test_size": 0.2,
        "inner_folds": int(grid["inner_folds"]),
        "selection_metric": grid["selection_metric"],
        "implementation": "benchscofi_2.0.1_official_models_with_label_neutral_prediction",
        "runtime_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "benchscofi": version("benchscofi"),
            "stanscofi": version("stanscofi"),
        },
        "counts": {
            "outer_train_pairs": int(outer_train_folds.nnz),
            "outer_test_pairs": int(outer_test_folds.nnz),
            "outer_train_positive": int((outer_labels == 1).sum()),
        },
        "leakage_audit": {
            "outer_train_test_overlap": 0,
            "outer_test_coordinates_seen_by_inner_cv": 0,
            "prediction_dataset_labels": "all_zero",
        },
        "input_sha256": {
            "ratings": sha256_file(ratings_path),
            "items": sha256_file(items_path),
            "users": sha256_file(users_path),
            "grid": sha256_file(grid_path),
        },
        "methods": methods,
        "runtime_sec": round(perf_counter() - started, 3),
        "interpretation": (
            "Hyperparameters are selected only within each outer training fold. "
            "Outer labels are unavailable to selection and prediction inputs contain zeros only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--grid", type=Path,
                        default=Path("configs/nested_cv_official_grid_v1.json"))
    parser.add_argument("--split", choices=["random_simple", "weakly_correlated"],
                        default="random_simple")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("benchmark/results"))
    args = parser.parse_args()
    report = run(args.data, args.grid, args.split, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"nested_cv_official_{args.split}_seed{args.seed}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
