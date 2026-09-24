"""RECeSS/stanscofi-compatible no-training baselines.

Only ``fit`` reads ratings, and only from the dataset supplied as its training
fold. ``predict_proba`` reads feature matrices and the availability mask, never
the validation ratings. This boundary is important because Dataset itself
contains held-out labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_array
from scipy.stats import rankdata

from .data import ExpressionData
from .ranking import _rrf, score_expressions


class TranscriptBaseline:
    def __init__(self, params: dict | None = None):
        params = params or {}
        self.method = params.get("method", "B1")
        if self.method not in {"B0", "B0p", "B1", "B1k", "B2"}:
            raise ValueError("Unknown baseline")
        self.seed = int(params.get("seed", 1234))
        self.neighbors = int(params.get("neighbors", 10))
        self.name = f"Transcript{self.method}"
        self._train_positive: np.ndarray | None = None
        self._drug_ids: list[str] = []
        self._disease_ids: list[str] = []

    def fit(self, train_dataset, seed: int | None = None):
        if seed is not None:
            self.seed = seed
        self._drug_ids = list(train_dataset.item_list)
        self._disease_ids = list(train_dataset.user_list)
        # A held-out fold is NaN after subset(); the ratings in the training
        # Dataset are therefore the only labels available to these baselines.
        positive = np.asarray(train_dataset.ratings.toarray() == 1, dtype=float)
        available = train_dataset.folds.toarray().astype(bool)
        self._train_positive = positive * available
        return self

    def _features(self, dataset) -> ExpressionData:
        if list(dataset.item_list) != self._drug_ids or list(dataset.user_list) != self._disease_ids:
            raise ValueError("Prediction IDs or order changed after fit")
        drugs = pd.DataFrame(dataset.items.toarray(), index=dataset.item_features,
                             columns=dataset.item_list)
        diseases = pd.DataFrame(dataset.users.toarray(), index=dataset.user_features,
                                columns=dataset.user_list)
        if set(drugs.index) != set(diseases.index):
            raise ValueError(f"Benchmark gene features diverged: drugs={len(drugs.index)}, "
                             f"diseases={len(diseases.index)}, "
                             f"drug_only={len(set(drugs.index) - set(diseases.index))}, "
                             f"disease_only={len(set(diseases.index) - set(drugs.index))}")
        return ExpressionData(drugs, diseases)

    def _knn(self, data: ExpressionData) -> np.ndarray:
        x = data.drugs.to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0)
        ranks = rankdata(x, axis=0)
        ranks -= ranks.mean(axis=0)
        norms = np.linalg.norm(ranks, axis=0)
        norms[norms == 0] = 1
        z = ranks / norms
        similarity = z.T @ z
        np.fill_diagonal(similarity, -np.inf)
        n = min(self.neighbors, similarity.shape[0] - 1)
        if n < 1:
            return np.zeros_like(self._train_positive)
        neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :n]
        weights = np.maximum(np.take_along_axis(similarity, neighbors, axis=1), 0)
        labels = self._train_positive[neighbors]
        return (weights[:, :, None] * labels).sum(axis=1) / np.maximum(weights.sum(axis=1)[:, None], 1e-12)

    def predict_proba(self, test_dataset) -> coo_array:
        if self._train_positive is None:
            raise RuntimeError("fit must be called before predict_proba")
        data = self._features(test_dataset)
        shape = self._train_positive.shape
        if self.method == "B0":
            values = np.random.default_rng(self.seed).random(shape)
        elif self.method == "B0p":
            values = np.broadcast_to(self._train_positive.sum(axis=1)[:, None], shape).copy()
        elif self.method == "B1":
            values = score_expressions(data)["spearman_reversal"].to_numpy()
        elif self.method == "B1k":
            values = self._knn(data)
        else:
            reversal = score_expressions(data)["spearman_reversal"].to_numpy()
            popularity = np.broadcast_to(self._train_positive.sum(axis=1)[:, None], shape)
            knn = self._knn(data)
            # Fixed RRF; every method ranks candidates within each disease.
            values = _rrf([popularity, knn, reversal])
        available = test_dataset.folds.tocoo()
        selected = values[available.row, available.col]
        # Keep all available pairs, including a legitimate zero score.
        return coo_array((selected, (available.row, available.col)), shape=shape)

    def predict(self, scores, threshold: float = 0):
        values = scores.toarray()
        available = scores.tocoo()
        labels = np.where(values[available.row, available.col] >= threshold, 1, -1)
        return coo_array((labels, (available.row, available.col)), shape=scores.shape)
