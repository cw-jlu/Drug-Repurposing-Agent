"""Check that the upstream adapter preserves the frozen B2 score calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stanscofi.models")

from benchmarks.recess_adapter.nested_cv import LocalDataset
from benchmarks.recess_adapter.official_b2 import B2
from benchmarks.recess_adapter.official_components import B0p, B1, B1k
from drug_repurposing_agent.benchmark import TranscriptBaseline


def test_official_b2_cached_predictions_match_frozen_baseline():
    genes = [f"g{i}" for i in range(8)]
    drugs = [f"d{i}" for i in range(4)]
    diseases = [f"c{i}" for i in range(3)]
    items = pd.DataFrame(np.arange(32, dtype=float).reshape(8, 4),
                         index=genes, columns=drugs)
    items.iloc[2] = [3, 1, 4, 2]
    users = pd.DataFrame(np.arange(24, dtype=float).reshape(8, 3)[::-1],
                         index=genes, columns=diseases)
    ratings = pd.DataFrame([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
                           index=drugs, columns=diseases)
    dataset = LocalDataset(ratings, items, users)
    prediction = dataset.subset(dataset.folds)
    for changed in (False, True):
        train = dataset.subset(dataset.folds)
        if changed:
            train.ratings.data[:] = 0
            train.ratings = train.ratings.tocoo()
        original = TranscriptBaseline({"method": "B2", "neighbors": 2,
                                       "rrf_k": 20}).fit(train)
        official = B2({"neighbors": 2, "rrf_k": 20}).fit(train)
        expected = original.predict_proba(prediction)
        actual = official.predict_proba(prediction)
        np.testing.assert_array_equal(actual.row, expected.row)
        np.testing.assert_array_equal(actual.col, expected.col)
        np.testing.assert_allclose(actual.data, expected.data, rtol=0, atol=0)
        np.testing.assert_allclose(official.predict_proba(prediction).data,
                                   expected.data, rtol=0, atol=0)
        changed_validation_labels = LocalDataset(-ratings, items, users,
                                                 folds=dataset.folds)
        label_blind = B2({"neighbors": 2, "rrf_k": 20}).fit(train)
        np.testing.assert_allclose(
            label_blind.predict_proba(changed_validation_labels).data,
            expected.data, rtol=0, atol=0
        )


@pytest.mark.parametrize("method,adapter", [("B0p", B0p), ("B1k", B1k), ("B1", B1)])
def test_official_components_match_frozen_baselines_without_validation_labels(method, adapter):
    genes = [f"g{i}" for i in range(8)]
    drugs = [f"d{i}" for i in range(4)]
    diseases = [f"c{i}" for i in range(3)]
    items = pd.DataFrame(np.arange(32, dtype=float).reshape(8, 4),
                         index=genes, columns=drugs)
    users = pd.DataFrame(np.arange(24, dtype=float).reshape(8, 3)[::-1],
                         index=genes, columns=diseases)
    ratings = pd.DataFrame([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
                           index=drugs, columns=diseases)
    dataset = LocalDataset(ratings, items, users)
    train = dataset.subset(dataset.folds)
    prediction = dataset.subset(dataset.folds)
    changed_labels = LocalDataset(-ratings, items, users, folds=dataset.folds)
    expected = TranscriptBaseline({"method": method, "neighbors": 2}).fit(train)
    actual = adapter({"neighbors": 2}).fit(train)
    reference = expected.predict_proba(prediction)
    np.testing.assert_array_equal(actual.predict_proba(prediction).row, reference.row)
    np.testing.assert_array_equal(actual.predict_proba(prediction).col, reference.col)
    np.testing.assert_allclose(actual.predict_proba(prediction).data, reference.data, rtol=0, atol=0)
    np.testing.assert_allclose(actual.predict_proba(changed_labels).data,
                               reference.data, rtol=0, atol=0)
