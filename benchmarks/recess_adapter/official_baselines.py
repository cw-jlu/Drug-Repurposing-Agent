"""Run three unmodified benchscofi algorithms on a fixed RECeSS split.

These are quick reference runs with package defaults, not the publication's
nested cross-validation or a claim about its published numerical results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
# benchscofi 2.0.1's PMF implementation still uses the removed np.int alias.
# This restores the historical meaning without modifying the model algorithm.
if not hasattr(np, "int"):
    np.int = int
import stanscofi.datasets
import stanscofi.training_testing
import stanscofi.validation

from benchscofi.ALSWR import ALSWR
from benchscofi.PMF import PMF
from benchscofi.LogisticMF import LogisticMF
from drug_repurposing_agent.data import sha256_file


MODELS = {"ALSWR": ALSWR, "PMF": PMF, "LogisticMF": LogisticMF}


def run(data_dir: Path, split: str, seed: int) -> dict:
    ratings_path = data_dir / "ratings_mat.csv"
    items_path = data_dir / "items.csv"
    users_path = data_dir / "users.csv"
    dataset = stanscofi.datasets.Dataset(
        ratings=pd.read_csv(ratings_path, index_col=0),
        items=pd.read_csv(items_path, index_col=0),
        users=pd.read_csv(users_path, index_col=0), name="TRANSCRIPT")
    splitter = getattr(stanscofi.training_testing, split + "_split")
    (train_folds, test_folds), _ = splitter(dataset, 0.2, metric="euclidean", random_state=seed)
    train = dataset.subset(train_folds)
    test = dataset.subset(test_folds)
    # Some upstream model preprocessors read dataset.ratings during prediction,
    # even though their fitted predictor ignores those values. Supply a
    # feature-identical, label-neutral test object to enforce the boundary.
    neutral = np.full(dataset.ratings.shape, np.nan)
    neutral[test_folds.row, test_folds.col] = 0
    test_features = stanscofi.datasets.Dataset(
        ratings=pd.DataFrame(neutral, index=dataset.item_list, columns=dataset.user_list),
        items=pd.DataFrame(dataset.items.toarray(), index=dataset.item_features,
                           columns=dataset.item_list),
        users=pd.DataFrame(dataset.users.toarray(), index=dataset.user_features,
                           columns=dataset.user_list), name="TRANSCRIPT_validation_features")
    truth = test.ratings.toarray()[test.folds.row, test.folds.col]
    labels = (truth == 1).astype(int)
    results = {"dataset": "TRANSCRIPT v2.0.0", "split": split, "seed": seed,
               "test_size": 0.2,
               "input_sha256": {"ratings": sha256_file(ratings_path),
                                "items": sha256_file(items_path),
                                "users": sha256_file(users_path)},
               "methods": {}}
    for name, model_class in MODELS.items():
        model = model_class()
        started = perf_counter()
        model.fit(train, seed=seed)
        score_matrix = model.predict_proba(test_features).toarray()
        selected = score_matrix[test.folds.row, test.folds.col]
        if not np.isfinite(selected).all():
            raise ValueError(f"{name} produced a non-finite prediction")
        auc = stanscofi.validation.AUC(labels, selected, 1, 1)
        ndcg = stanscofi.validation.NDCGk(labels, selected, len(selected), 1)
        parameters = {key: value for key, value in model.default_parameters().items()
                      if key != "counts"}
        results["methods"][name] = {"global_AUC": float(auc),
                                     "global_NDCG": float(ndcg),
                                     "runtime_sec": round(perf_counter() - started, 3),
                                     "parameters": parameters}
        print(f"{split} seed={seed} {name}: AUC={auc:.4f} NDCG={ndcg:.4f}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["random_simple", "weakly_correlated"],
                        default="random_simple")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, default=Path("benchmark/results"))
    args = parser.parse_args()
    result = run(args.data, args.split, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"recess_defaults_{args.split}_seed{args.seed}.json"
    path.write_text(json.dumps(result, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x),
                    encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
