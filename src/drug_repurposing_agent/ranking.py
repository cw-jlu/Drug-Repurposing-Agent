"""Fixed, label-free transcriptomic scoring methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import ExpressionData


def _rank_z(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, axis=0)
    ranks -= ranks.mean(axis=0)
    scale = np.sqrt(np.sum(ranks * ranks, axis=0))
    scale[scale == 0] = 1
    return ranks / scale


def _pairwise_spearman(drugs: np.ndarray, diseases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Negative Spearman and valid-gene counts; undefined pairs get zero."""
    if np.isfinite(drugs).all() and np.isfinite(diseases).all():
        scores = -_rank_z(drugs).T @ _rank_z(diseases)
        return scores, np.full(scores.shape, drugs.shape[0], dtype=int)
    scores = np.zeros((drugs.shape[1], diseases.shape[1]), dtype=float)
    counts = np.zeros_like(scores, dtype=int)
    for i in range(drugs.shape[1]):
        for j in range(diseases.shape[1]):
            valid = np.isfinite(drugs[:, i]) & np.isfinite(diseases[:, j])
            counts[i, j] = valid.sum()
            if counts[i, j] < 3:
                continue
            x, y = drugs[valid, i], diseases[valid, j]
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                continue
            scores[i, j] = -(_rank_z(x[:, None]).T @ _rank_z(y[:, None]))[0, 0]
    return scores, counts


def _connectivity(drugs: np.ndarray, diseases: np.ndarray, top_k: int) -> np.ndarray:
    # Reversal: disease up should sit low in the drug ranked signature.
    drug_ranks = np.apply_along_axis(lambda x: rankdata(np.nan_to_num(x, nan=0.0)), 0, drugs)
    drug_ranks = (drug_ranks - (len(drugs) + 1) / 2) / max(len(drugs) / 2, 1)
    out = np.zeros((drugs.shape[1], diseases.shape[1]), dtype=float)
    for j in range(diseases.shape[1]):
        valid = np.flatnonzero(np.isfinite(diseases[:, j]))
        if len(valid) < 2 or np.ptp(diseases[valid, j]) == 0:
            continue
        k = min(top_k, len(valid) // 2)
        order = valid[np.argsort(diseases[valid, j], kind="stable")]
        down, up = order[:k], order[-k:]
        out[:, j] = (drug_ranks[down].mean(axis=0) - drug_ranks[up].mean(axis=0)) / 2
    return out


def _rrf(scores: list[np.ndarray], k: int = 60) -> np.ndarray:
    fused = np.zeros_like(scores[0])
    for score in scores:
        # Rank 1 is best; tied scores share a rank rather than inheriting
        # the input file's row order as an implicit prior.
        ranks = rankdata(-score, axis=0, method="average")
        fused += 1 / (k + ranks)
    return fused


def connectivity_gene_sets(drugs: pd.DataFrame, up: set[str], down: set[str]) -> pd.Series:
    """Score opposition to prespecified disease up/down sets for each drug."""
    if not up or not down or up & down:
        raise ValueError("Up/down gene sets must be nonempty and disjoint")
    if not up.issubset(drugs.index) or not down.issubset(drugs.index):
        raise ValueError("Disease gene sets contain genes absent from drug matrix")
    values = drugs.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Connectivity gene-set scoring requires finite values")
    ranks = rankdata(values, axis=0)
    ranks = (ranks - (len(drugs) + 1) / 2) / max(len(drugs) / 2, 1)
    up_idx = drugs.index.get_indexer(sorted(up))
    down_idx = drugs.index.get_indexer(sorted(down))
    scores = (ranks[down_idx].mean(axis=0) - ranks[up_idx].mean(axis=0)) / 2
    return pd.Series(scores, index=drugs.columns, name="connectivity")


def score_expressions(data: ExpressionData, top_k: int = 100) -> dict[str, pd.DataFrame]:
    drugs = data.drugs.to_numpy(dtype=float)
    diseases = data.diseases.to_numpy(dtype=float)
    spearman, counts = _pairwise_spearman(drugs, diseases)
    connectivity = _connectivity(drugs, diseases, top_k)
    fused = _rrf([spearman, connectivity])
    constant_diseases = np.array([np.nanstd(diseases[:, j]) == 0
                                  for j in range(diseases.shape[1])])
    fused[:, constant_diseases] = 0
    wrap = lambda x: pd.DataFrame(x, index=data.drugs.columns, columns=data.diseases.columns)
    return {"spearman_reversal": wrap(spearman), "connectivity": wrap(connectivity),
            "rrf": wrap(fused), "gene_counts": wrap(counts)}
