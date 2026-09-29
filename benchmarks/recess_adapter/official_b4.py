"""B4: within-drug-row rank ensemble of B3 and the NumPy BNNR port.

Both members are fitted on the same training fold handed over by the upstream
runner (training-fold ratings plus expression features only):

* ``B3``   frozen row-rank fusion (``official_b3.row_fusion_scores``);
* ``BNNR`` ``bnnr_numpy.BNNRNumpy`` (benchscofi preprocessing + MATLAB port).

Ensemble modes (``spec``):

* ``"block"``: ``(w_b3 * rowrank(B3) + w_bnnr * rowrank(BNNR)) / (w_b3 + w_bnnr)``;
* ``"flat"``:  BNNR joins the six B3 components as a seventh equally weighted
  row rank, ``(6 * mean_rank_B3 + rowrank(BNNR)) / 7``.

A ``1e-6`` x within-row rank of training disease popularity breaks ties, as in
B3.  The frozen spec (``FROZEN_SPEC``) was selected on random_simple
development seeds disjoint from the 100 official seeds; see
``configs/b4_ensemble_v1.json``.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array
from stanscofi.models import BasicModel

from .bnnr_numpy import BNNRNumpy
from .official_b3 import ALL_COMPONENTS, B3, _row_rank

FROZEN_SPEC = {"mode": "block", "b3_weight": 1.0, "bnnr_weight": 2.0}


def ensemble_scores(b3_scores: np.ndarray, bnnr_scores: np.ndarray,
                    train_positive: np.ndarray, spec: dict) -> np.ndarray:
    """Combine full drugs x diseases B3 and BNNR score matrices."""
    b3_scores = np.asarray(b3_scores, dtype=float)
    bnnr_scores = np.asarray(bnnr_scores, dtype=float)
    if b3_scores.shape != bnnr_scores.shape or b3_scores.shape != train_positive.shape:
        raise ValueError("B3, BNNR and training matrices must share a shape")
    popularity = np.broadcast_to(np.asarray(train_positive, dtype=float).sum(
        axis=0, keepdims=True), b3_scores.shape)
    tie_breaker = 1e-6 * _row_rank(popularity)
    bnnr_rank = _row_rank(bnnr_scores)
    mode = spec.get("mode")
    if mode == "block":
        w_b3, w_bnnr = float(spec["b3_weight"]), float(spec["bnnr_weight"])
        if w_b3 < 0 or w_bnnr < 0 or w_b3 + w_bnnr <= 0:
            raise ValueError("Ensemble weights must be non-negative and not both zero")
        fused = (w_b3 * _row_rank(b3_scores) + w_bnnr * bnnr_rank) / (w_b3 + w_bnnr)
    elif mode == "flat":
        mean_rank = b3_scores - tie_breaker  # B3 = mean component rank + tie-breaker
        n = len(ALL_COMPONENTS)
        fused = (n * mean_rank + bnnr_rank) / (n + 1)
    else:
        raise ValueError(f"Unknown B4 ensemble mode: {mode!r}")
    return fused + tie_breaker


class B4(BasicModel):
    spec: dict = FROZEN_SPEC
    model_name = "B4"

    def __init__(self, params: dict | None = None):
        params = dict(params or {})
        spec = params.pop("spec", None)
        super().__init__(params)
        self.name = self.model_name
        self.spec = dict(spec if spec is not None else type(self).spec)
        self.b3 = B3()
        self.bnnr = BNNRNumpy()
        self._b3_scores: np.ndarray | None = None
        self._bnnr_scores: np.ndarray | None = None
        self._score_values: np.ndarray | None = None

    def fit(self, train_dataset, seed: int = 1234):
        self._score_values = None
        self.b3.fit(train_dataset, seed=seed)
        self.b3.predict_proba(train_dataset)  # materialises the full B3 matrix
        self._b3_scores = self.b3._score_values.copy()
        self.bnnr.fit(train_dataset, seed=seed)
        self._bnnr_scores = self.bnnr.estimator["predictions"].copy()
        return self

    def set_spec(self, spec: dict) -> None:
        self.spec = dict(spec)
        self._score_values = None

    def predict_proba(self, test_dataset) -> coo_array:
        if self._b3_scores is None or self._bnnr_scores is None:
            raise RuntimeError("fit must be called before predict_proba")
        if self._score_values is None:
            self._score_values = ensemble_scores(self._b3_scores, self._bnnr_scores,
                                                 self.b3._train_positive, self.spec)
        available = test_dataset.folds.tocoo()
        selected = self._score_values[available.row, available.col]
        return coo_array((selected, (available.row, available.col)),
                         shape=self._score_values.shape)
