"""The three individually scored components of the frozen B2 model.

These adapters reuse B2's feature-only cache and the official runner's fold
selection.  They do not read validation ratings when producing predictions.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array

from .official_b2 import B2


class _Component(B2):
    component: str

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.name = self.component

    def predict_proba(self, test_dataset) -> coo_array:
        if self._train_positive is None:
            raise RuntimeError("fit must be called before predict_proba")
        if self._score_values is None:
            if self.component == "B0p":
                self._score_values = np.broadcast_to(
                    self._train_positive.sum(axis=1)[:, None],
                    self._train_positive.shape,
                ).copy()
            else:
                reversal, neighbors, weights = self._feature_components(test_dataset)
                if self.component == "B1":
                    self._score_values = reversal
                else:
                    labels = self._train_positive[neighbors]
                    self._score_values = (weights[:, :, None] * labels).sum(axis=1) / np.maximum(
                        weights.sum(axis=1)[:, None], 1e-12
                    )
        # Preserve the frozen B2 adapter's ID-order guard even for B0p.
        if self.component == "B0p":
            self._features(test_dataset)
        available = test_dataset.folds.tocoo()
        selected = self._score_values[available.row, available.col]
        return coo_array((selected, (available.row, available.col)),
                         shape=self._train_positive.shape)


class B0p(_Component):
    component = "B0p"


class B1k(_Component):
    component = "B1k"


class B1(_Component):
    component = "B1"
