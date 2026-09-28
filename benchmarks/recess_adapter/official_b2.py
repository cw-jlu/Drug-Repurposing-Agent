"""B2 model adapter for the upstream RECeSS benchmark pipeline.

The upstream runner expects a class taking ``params`` and exposing the
``stanscofi.BasicModel`` reporting methods.  The scientific calculation stays
in the versioned TranscriptBaseline implementation used by this repository.
Only training-fold ratings are read in ``fit``; prediction uses features and
fold coordinates, not validation ratings.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array
from scipy.stats import rankdata
from stanscofi.models import BasicModel

from drug_repurposing_agent.benchmark import TranscriptBaseline
from drug_repurposing_agent.ranking import _pairwise_spearman, _rrf


class B2(TranscriptBaseline, BasicModel):
    """Expose fixed B2 parameters to the official five-fold model selector."""

    _feature_cache: dict | None = None

    def __init__(self, params: dict | None = None):
        supplied = dict(params or {})
        if supplied.get("method", "B2") != "B2":
            raise ValueError("The official B2 adapter only accepts method=B2")
        supplied["method"] = "B2"
        super().__init__(supplied)
        self.name = "B2"
        self._score_values: np.ndarray | None = None

    def fit(self, train_dataset, seed: int | None = None):
        self._score_values = None
        return super().fit(train_dataset, seed=seed)

    def _feature_components(self, dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = self._features(dataset)
        drugs = data.drugs.to_numpy(dtype=float)
        diseases = data.diseases.to_numpy(dtype=float)
        cache = type(self)._feature_cache
        if (cache is None or cache["neighbors_n"] != self.neighbors
                or not np.array_equal(drugs, cache["drugs"], equal_nan=True)
                or not np.array_equal(diseases, cache["diseases"], equal_nan=True)):
            reversal = _pairwise_spearman(drugs, diseases)[0]
            x = np.nan_to_num(drugs, nan=0.0)
            ranks = rankdata(x, axis=0)
            ranks -= ranks.mean(axis=0)
            norms = np.linalg.norm(ranks, axis=0)
            norms[norms == 0] = 1
            z = ranks / norms
            similarity = z.T @ z
            np.fill_diagonal(similarity, -np.inf)
            neighbors_n = min(self.neighbors, similarity.shape[0] - 1)
            neighbors = np.argsort(-similarity, axis=1, kind="stable")[:, :neighbors_n]
            weights = np.maximum(np.take_along_axis(similarity, neighbors, axis=1), 0)
            cache = {"drugs": drugs.copy(), "diseases": diseases.copy(),
                     "neighbors_n": self.neighbors, "reversal": reversal,
                     "neighbors": neighbors, "weights": weights}
            type(self)._feature_cache = cache
        return cache["reversal"], cache["neighbors"], cache["weights"]

    def predict_proba(self, test_dataset) -> coo_array:
        if self._train_positive is None:
            raise RuntimeError("fit must be called before predict_proba")
        if self._score_values is None:
            reversal, neighbors, weights = self._feature_components(test_dataset)
            popularity = np.broadcast_to(self._train_positive.sum(axis=1)[:, None],
                                         self._train_positive.shape)
            labels = self._train_positive[neighbors]
            knn = (weights[:, :, None] * labels).sum(axis=1) / np.maximum(
                weights.sum(axis=1)[:, None], 1e-12
            )
            self._score_values = _rrf([popularity, knn, reversal], k=self.rrf_k)
        available = test_dataset.folds.tocoo()
        selected = self._score_values[available.row, available.col]
        return coo_array((selected, (available.row, available.col)),
                         shape=self._train_positive.shape)
