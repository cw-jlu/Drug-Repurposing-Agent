import numpy as np
import pytest

from benchmarks.recess_adapter.official_b3 import ALL_COMPONENTS, row_fusion_scores


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    genes, drugs, diseases = 40, 12, 7
    y = (rng.random((drugs, diseases)) < 0.2).astype(float)
    return y, rng.normal(size=(genes, drugs)), rng.normal(size=(genes, diseases))


def test_scores_have_expected_shape_and_are_finite():
    y, xd, xs = _toy()
    scores = row_fusion_scores(y, xd, xs)
    assert scores.shape == y.shape
    assert np.isfinite(scores).all()


def test_disease_popularity_varies_within_drug_rows():
    y, xd, xs = _toy()
    scores = row_fusion_scores(y, xd, xs, ("DP",))
    order = np.argsort(y.sum(axis=0))
    assert np.all(np.diff(scores[0, order]) >= 0)
    assert np.ptp(scores[0]) > 0


def test_tie_breaker_removes_all_zero_rows():
    y, xd, xs = _toy()
    y[3] = 0
    scores = row_fusion_scores(y, xd, xs, ("LD",))
    assert len(np.unique(scores[3])) > 1


def test_scores_depend_only_on_supplied_training_labels():
    y, xd, xs = _toy()
    first = row_fusion_scores(y, xd, xs)
    second = row_fusion_scores(y.copy(), xd.copy(), xs.copy())
    assert np.array_equal(first, second)
    changed = y.copy()
    changed[0, 0] = 1 - changed[0, 0]
    assert not np.array_equal(first, row_fusion_scores(changed, xd, xs))


def test_unknown_component_rejected():
    y, xd, xs = _toy()
    with pytest.raises(ValueError):
        row_fusion_scores(y, xd, xs, ("DP", "BOGUS"))
    assert set(ALL_COMPONENTS) == {"DP", "DK", "SK", "LD", "LS", "REV"}
