"""Nested-CV tuning for the label-aware TRANSCRIPT baselines.

The outer test fold is created once and is never passed to inner selection.
Unknown ratings follow the official RECeSS global-metric convention and count
as non-positive; this is an evaluation convention, not a clinical label.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from itertools import chain
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import coo_array
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import sklearn

from drug_repurposing_agent.benchmark import TranscriptBaseline
from drug_repurposing_agent.data import sha256_file


class LocalDataset:
    """Small stanscofi-compatible boundary used to avoid optional plotting deps."""

    def __init__(self, ratings: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame,
                 folds=None, name: str = "TRANSCRIPT"):
        self.name = name
        self.item_list = list(ratings.index)
        self.user_list = list(ratings.columns)
        self.item_features = list(items.index)
        self.user_features = list(users.index)
        self.ratings = coo_array(ratings.to_numpy(dtype=float))
        self.items = coo_array(items.loc[:, self.item_list].to_numpy(dtype=float))
        self.users = coo_array(users.loc[:, self.user_list].to_numpy(dtype=float))
        if folds is None:
            available = np.argwhere(np.isfinite(ratings.to_numpy(dtype=float)))
            row, col = available[:, 0], available[:, 1]
            folds = coo_array((np.ones(len(row)), (row, col)), shape=ratings.shape)
        self.folds = folds.tocoo()

    def subset(self, folds):
        values = np.full(self.folds.shape, np.nan)
        source = self.ratings.toarray()
        selected = folds.tocoo()
        values[selected.row, selected.col] = source[selected.row, selected.col]
        ratings = pd.DataFrame(values, index=self.item_list, columns=self.user_list)
        items = pd.DataFrame(self.items.toarray(), index=self.item_features,
                             columns=self.item_list)
        users = pd.DataFrame(self.users.toarray(), index=self.user_features,
                             columns=self.user_list)
        return LocalDataset(ratings, items, users, selected, self.name)


def indices_to_folds(indices, coordinates, shape):
    selected = coordinates[indices]
    return coo_array((np.ones(len(selected)), (selected[:, 0], selected[:, 1])),
                     shape=shape)


def random_cv_split(dataset, cv):
    coordinates = np.column_stack((dataset.folds.row, dataset.folds.col))
    labels = dataset.ratings.toarray()[dataset.folds.row, dataset.folds.col]
    return [(indices_to_folds(train, coordinates, dataset.folds.shape),
             indices_to_folds(validation, coordinates, dataset.folds.shape))
            for train, validation in cv.split(coordinates, labels)]


def random_simple_split(dataset, test_size: float, seed: int):
    coordinates = np.column_stack((dataset.folds.row, dataset.folds.col))
    labels = dataset.ratings.toarray()[dataset.folds.row, dataset.folds.col]
    train, test, _, _ = train_test_split(
        coordinates, labels, test_size=test_size, random_state=seed,
        shuffle=True, stratify=labels)
    make = lambda selected: coo_array(
        (np.ones(len(selected)), (selected[:, 0], selected[:, 1])),
        shape=dataset.folds.shape)
    return make(train), make(test)


def weakly_correlated_split(dataset, test_size: float, seed: int):
    """Exact split logic used by stanscofi 2.0.1 for the required inputs."""
    del seed  # stanscofi's clustering path is deterministic for fixed features.
    item_matrix = np.nan_to_num(dataset.items.toarray(), copy=True, nan=0)
    train_nset = int((1 - test_size) * dataset.folds.data.shape[0])
    distances = pairwise_distances(item_matrix.T, metric="euclidean")
    hierarchy = linkage(squareform(distances, checks=False), "average")
    lower, upper = 2, item_matrix.shape[1]
    iteration = 0
    while lower < upper:
        clusters_n = (lower + upper) // 2
        clusters = fcluster(hierarchy, clusters_n, criterion="maxclust", depth=2)
        drug_clusters = clusters[dataset.folds.row]
        rating_counts = {
            drug_clusters[drug_clusters <= cutoff].shape[0]: cutoff
            for cutoff in range(1, len(np.unique(clusters)) + 1)
        }
        selected_count = max(k if k <= train_nset else -1 for k in rating_counts)
        cluster_cutoff = rating_counts.get(selected_count, -1)
        if selected_count == train_nset:
            break
        if selected_count < train_nset:
            lower = clusters_n + 1
        else:
            upper = clusters_n
        iteration += 1
        if iteration >= 100:
            break
    labels = (fcluster(hierarchy, clusters_n, criterion="maxclust", depth=2)
              > cluster_cutoff + 1).astype(int) + 1
    train = dataset.folds.toarray()
    train[labels == 2, :] = 0
    test = dataset.folds.toarray() - train
    return coo_array(train), coo_array(test)


def metric_values(dataset, scores) -> dict[str, float]:
    truth = dataset.ratings.toarray()[dataset.folds.row, dataset.folds.col]
    labels = (truth == 1).astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("Scoring fold has only one binary class")
    ranked = labels[np.argsort(-scores.data)]
    ideal = np.sort(labels)[::-1]
    def dcg(values):
        values = np.asarray(values, dtype=float)
        return values[0] + np.sum(values[1:] / np.log2(np.arange(2, len(values) + 1)))
    ideal_dcg = dcg(ideal)
    return {
        "global_AUC": float(roc_auc_score(labels, scores.data)),
        "global_NDCG": float(dcg(ranked) / ideal_dcg if ideal_dcg else 0.0),
    }


def fold_coordinates(folds) -> set[tuple[int, int]]:
    return set(zip(folds.row.tolist(), folds.col.tolist()))


def evaluate_candidate(method: str, params: dict, outer_train, inner_splits,
                       seed: int) -> dict:
    fold_scores = []
    for fold_index, (train_folds, validation_folds) in enumerate(inner_splits):
        train = outer_train.subset(train_folds)
        validation = outer_train.subset(validation_folds)
        model_params = {"method": method, "seed": seed, **params}
        model = TranscriptBaseline(model_params).fit(train, seed=seed)
        scores = model.predict_proba(validation)
        values = metric_values(validation, scores)
        fold_scores.append({"fold": fold_index, **values,
                            "train_pairs": int(train_folds.nnz),
                            "validation_pairs": int(validation_folds.nnz)})
    return {
        "parameters": params,
        "inner_folds": fold_scores,
        "mean_global_AUC": float(np.mean([x["global_AUC"] for x in fold_scores])),
        "mean_global_NDCG": float(np.mean([x["global_NDCG"] for x in fold_scores])),
    }


def run(data_dir: Path, grid_path: Path, split: str, seed: int) -> dict:
    started = perf_counter()
    ratings_path = data_dir / "ratings_mat.csv"
    items_path = data_dir / "items.csv"
    users_path = data_dir / "users.csv"
    dataset = LocalDataset(
        ratings=pd.read_csv(ratings_path, index_col=0),
        items=pd.read_csv(items_path, index_col=0),
        users=pd.read_csv(users_path, index_col=0), name="TRANSCRIPT")
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    if split == "random_simple":
        outer_train_folds, outer_test_folds = random_simple_split(dataset, 0.2, seed)
    else:
        outer_train_folds, outer_test_folds = weakly_correlated_split(dataset, 0.2, seed)
    if (outer_train_folds.multiply(outer_test_folds)).nnz:
        raise RuntimeError("Outer training and test folds overlap")

    outer_train = dataset.subset(outer_train_folds)
    outer_test = dataset.subset(outer_test_folds)
    cv = StratifiedKFold(n_splits=int(grid["inner_folds"]), shuffle=True,
                         random_state=seed + 10000)
    inner_splits = random_cv_split(outer_train, cv)
    outer_test_coords = fold_coordinates(outer_test_folds)
    inner_coords = set(chain.from_iterable(
        fold_coordinates(train) | fold_coordinates(validation)
        for train, validation in inner_splits))
    if outer_test_coords & inner_coords:
        raise RuntimeError("Outer test coordinates leaked into inner CV")

    methods = {}
    for method, candidates in grid["methods"].items():
        evaluated = []
        for params in candidates:
            result = evaluate_candidate(method, params, outer_train, inner_splits, seed)
            evaluated.append(result)
            print(f"{split} seed={seed} {method} {params}: "
                  f"inner AUC={result['mean_global_AUC']:.4f}", flush=True)
        selected = sorted(
            evaluated,
            key=lambda x: (-x["mean_global_AUC"], -x["mean_global_NDCG"],
                           json.dumps(x["parameters"], sort_keys=True)),
        )[0]
        model = TranscriptBaseline({"method": method, "seed": seed,
                                    **selected["parameters"]}).fit(outer_train, seed=seed)
        outer_scores = model.predict_proba(outer_test)
        methods[method] = {
            "selected_parameters": selected["parameters"],
            "selection": {
                "metric": grid["selection_metric"],
                "mean_global_AUC": selected["mean_global_AUC"],
                "mean_global_NDCG": selected["mean_global_NDCG"],
            },
            "outer_test": metric_values(outer_test, outer_scores),
            "candidate_results": evaluated,
        }
        print(f"selected {method} {selected['parameters']}: "
              f"outer AUC={methods[method]['outer_test']['global_AUC']:.4f}", flush=True)

    return {
        "protocol": grid["name"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "TRANSCRIPT v2.0.0",
        "split": split,
        "seed": seed,
        "outer_test_size": 0.2,
        "inner_folds": int(grid["inner_folds"]),
        "selection_metric": grid["selection_metric"],
        "implementation": "stanscofi_2.0.1_split_and_metric_equivalent_lightweight_runner",
        "runtime_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "pandas": pd.__version__, "scipy": scipy.__version__,
                             "scikit_learn": sklearn.__version__},
        "counts": {"outer_train_pairs": int(outer_train_folds.nnz),
                   "outer_test_pairs": int(outer_test_folds.nnz)},
        "leakage_audit": {
            "outer_train_test_overlap": 0,
            "outer_test_coordinates_seen_by_inner_cv": 0,
            "inner_coordinate_union": len(inner_coords),
        },
        "input_sha256": {"ratings": sha256_file(ratings_path),
                         "items": sha256_file(items_path),
                         "users": sha256_file(users_path),
                         "grid": sha256_file(grid_path)},
        "methods": methods,
        "runtime_sec": round(perf_counter() - started, 3),
        "interpretation": ("Unknown pairs are non-positive only under the official global "
                           "metric convention; selected parameters use outer-training data only."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--grid", type=Path, default=Path("configs/nested_cv_grid_v1.json"))
    parser.add_argument("--split", choices=["random_simple", "weakly_correlated"],
                        default="random_simple")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("benchmark/results"))
    args = parser.parse_args()
    report = run(args.data, args.grid, args.split, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"nested_cv_{args.split}_seed{args.seed}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
