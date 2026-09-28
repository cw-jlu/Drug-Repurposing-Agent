"""NumPy BNNR port and B4 ensemble adapter on tiny matrices."""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stanscofi.models")
pytest.importorskip("benchscofi")
np.asfarray = getattr(np, "asfarray", lambda v: np.asarray(v, dtype=float))

import stanscofi.datasets
from scipy.sparse import coo_array

from benchmarks.recess_adapter.bnnr_numpy import BNNRNumpy, bnnr, svt



def _dataset(seed=0, drugs=9, diseases=6, genes=15):
    rng = np.random.default_rng(seed)
    drug_ids = [f"d{i}" for i in range(drugs)]
    disease_ids = [f"c{i}" for i in range(diseases)]
    gene_ids = [f"g{i}" for i in range(genes)]
    ratings = np.zeros((drugs, diseases))
    ratings[rng.random(ratings.shape) < 0.3] = 1
    ratings[rng.random(ratings.shape) < 0.05] = -1
    ratings[0, 0], ratings[1, 1] = 1, 1
    return stanscofi.datasets.Dataset(
        ratings=pd.DataFrame(ratings, index=drug_ids, columns=disease_ids),
        items=pd.DataFrame(rng.normal(size=(genes, drugs)), index=gene_ids, columns=drug_ids),
        users=pd.DataFrame(rng.normal(size=(genes, diseases)), index=gene_ids,
                           columns=disease_ids),
        same_item_user_features=True, name="toy")


def _split(dataset, seed=1):
    rng = np.random.default_rng(seed)
    mask = rng.random(dataset.folds.shape) < 0.7
    train = coo_array(mask.astype(float))
    test = coo_array((~mask).astype(float))
    return dataset.subset(train), dataset.subset(test), mask


def test_svt_matches_definition():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(7, 4))
    u, s, vt = np.linalg.svd(y, full_matrices=False)
    expected = u @ np.diag(np.maximum(s - 1.0, 0)) @ vt
    np.testing.assert_allclose(svt(y, 1.0), expected, atol=1e-12)
    np.testing.assert_allclose(svt(y, s.max() + 1), 0, atol=1e-12)


def test_bnnr_respects_bounds_and_recovers_low_rank():
    rng = np.random.default_rng(1)
    truth = np.clip(rng.random((12, 1)) @ rng.random((1, 12)), 0, 1)
    observed = (rng.random(truth.shape) < 0.7).astype(float)
    recovered, n_iter = bnnr(1, 10, truth * observed, observed, 2e-3, 1e-5, 300, 0, 1)
    assert recovered.shape == truth.shape and 0 < n_iter <= 300
    assert recovered.min() >= 0 and recovered.max() <= 1
    hidden = observed == 0
    assert np.abs(recovered - truth)[hidden].mean() < np.abs(truth)[hidden].mean()


def test_bnnr_numpy_shapes_determinism_and_defaults():
    dataset = _dataset()
    train, test, _ = _split(dataset)
    first, second = BNNRNumpy(), BNNRNumpy()
    assert (first.alpha, first.beta, first.tol1, first.tol2, first.maxiter) == (1, 10, 2e-3, 1e-5, 300)
    first.fit(train); second.fit(train)
    assert first.estimator["predictions"].shape == dataset.folds.shape
    np.testing.assert_array_equal(first.estimator["predictions"], second.estimator["predictions"])
    scores = first.predict_proba(test)
    assert scores.shape == dataset.folds.shape


def test_bnnr_numpy_label_isolation():
    dataset = _dataset()
    train, _, mask = _split(dataset)
    base = BNNRNumpy(); base.fit(train)
    flipped = dataset.ratings.toarray().copy()
    flipped[~mask] = np.where(flipped[~mask] == 1, 0, 1)  # change held-out labels only
    other = stanscofi.datasets.Dataset(
        ratings=pd.DataFrame(flipped, index=dataset.item_list, columns=dataset.user_list),
        items=pd.DataFrame(dataset.items.toarray(), index=dataset.item_features,
                           columns=dataset.item_list),
        users=pd.DataFrame(dataset.users.toarray(), index=dataset.user_features,
                           columns=dataset.user_list),
        same_item_user_features=True, name="toy2")
    other_train = other.subset(coo_array(mask.astype(float)))
    model = BNNRNumpy(); model.fit(other_train)
    np.testing.assert_array_equal(base.estimator["predictions"], model.estimator["predictions"])
    changed = train.ratings.toarray().copy()
    changed[mask] = np.where(changed[mask] == 1, 0, 1)
    assert not np.array_equal(changed, train.ratings.toarray())
