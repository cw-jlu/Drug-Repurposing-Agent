"""B3: row-oriented, training-free fusion for the upstream RECeSS runner.

The official NS-AUC ("Lin's AUC") ranks diseases *within each drug row* and
scores ties as 0 (strict ``>``).  B2 ranked within disease columns and used
drug popularity, which is constant within a row.  B3 fixes the orientation:
every component is rank-normalised within the drug row, then averaged, with a
deterministic disease-popularity tie-breaker.

Components (training-fold labels ``Y`` and expression features only; no
parameters are fitted):

* ``DP``  disease popularity: column sums of ``Y``;
* ``DK``  drug-expression kNN: neighbours' ``Y`` rows (k=10, Spearman);
* ``SK``  disease-expression kNN: neighbours' ``Y`` columns (k=10);
* ``LD``  label co-occurrence drug CF: cosine of ``Y`` rows, k=20;
* ``LS``  label co-occurrence disease CF: ``Y @ cosine(Y columns)``;
* ``REV`` negative Spearman signature reversal (B1).

The configuration is frozen in ``configs/b3_row_fusion_v1.json`` and was
chosen on eight development seeds disjoint from the 100 official seeds.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array
from scipy.stats import rankdata

from .official_b2 import B2

ALL_COMPONENTS = ("DP", "DK", "SK", "LD", "LS", "REV")


def _zcols(matrix: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.nan_to_num(matrix, nan=0.0), axis=0)
    ranks -= ranks.mean(axis=0)
    norms = np.linalg.norm(ranks, axis=0)
    norms[norms == 0] = 1
    return ranks / norms


def _knn_propagate(similarity: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    similarity = similarity.copy()
    np.fill_diagonal(similarity, -np.inf)
    k = min(k, similarity.shape[0] - 1)
    idx = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    weights = np.maximum(np.take_along_axis(similarity, idx, axis=1), 0)
    return (weights[:, :, None] * labels[idx]).sum(axis=1) / np.maximum(
        weights.sum(axis=1)[:, None], 1e-12)


def _cosine(labels: np.ndarray) -> np.ndarray:
    gram = labels @ labels.T
    norms = np.sqrt(np.outer(labels.sum(axis=1), labels.sum(axis=1)))
    return np.divide(gram, norms, out=np.zeros_like(gram), where=norms > 0)


def _row_rank(matrix: np.ndarray) -> np.ndarray:
    return rankdata(matrix, axis=1) / matrix.shape[1]


def row_fusion_scores(train_positive: np.ndarray, drug_features: np.ndarray,
                      disease_features: np.ndarray,
                      components: tuple[str, ...] = ALL_COMPONENTS) -> np.ndarray:
    """Return the drugs x diseases B3 score matrix (features are genes x entities)."""
    unknown = set(components) - set(ALL_COMPONENTS)
    if not components or unknown:
        raise ValueError(f"Unknown B3 components: {sorted(unknown)}")
    y = np.asarray(train_positive, dtype=float)
    zd, zs = _zcols(drug_features), _zcols(disease_features)
    built = {}
    if "DP" in components:
        built["DP"] = np.broadcast_to(y.sum(axis=0, keepdims=True), y.shape).copy()
    if "DK" in components:
        built["DK"] = _knn_propagate(zd.T @ zd, y, 10)
    if "SK" in components:
        built["SK"] = _knn_propagate(zs.T @ zs, y.T, 10).T
    if "LD" in components:
        built["LD"] = _knn_propagate(_cosine(y), y, 20)
    if "LS" in components:
        built["LS"] = y @ _cosine(y.T)
    if "REV" in components:
        built["REV"] = -(zd.T @ zs)
    fused = sum(_row_rank(built[name]) for name in components) / len(components)
    tie_breaker = 1e-6 * _row_rank(np.broadcast_to(y.sum(axis=0, keepdims=True), y.shape))
    return fused + tie_breaker


class B3(B2):
    components: tuple[str, ...] = ALL_COMPONENTS
    model_name = "B3"

    def __init__(self, params: dict | None = None):
        super().__init__({k: v for k, v in (params or {}).items() if k != "components"})
        self.name = self.model_name

    def predict_proba(self, test_dataset) -> coo_array:
        if self._train_positive is None:
            raise RuntimeError("fit must be called before predict_proba")
        if self._score_values is None:
            data = self._features(test_dataset)
            drugs = data.drugs.to_numpy(dtype=float)
            diseases = data.diseases.loc[data.drugs.index].to_numpy(dtype=float)
            self._score_values = row_fusion_scores(self._train_positive, drugs, diseases,
                                                   self.components)
        available = test_dataset.folds.tocoo()
        selected = self._score_values[available.row, available.col]
        return coo_array((selected, (available.row, available.col)),
                         shape=self._train_positive.shape)


class B3noREV(B3):
    """Post-hoc variant without REV; disclosed as chosen after the weak holdout was seen."""

    components = ("DP", "DK", "SK", "LD", "LS")
    model_name = "B3noREV"
