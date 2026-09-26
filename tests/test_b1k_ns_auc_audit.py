import numpy as np
from scipy.sparse import coo_array

from evals.audit_b1k_ns_auc import pair_counts


def test_pair_counts_exposes_ties_and_direction():
    folds = coo_array((np.ones(3), ([0, 0, 0], [0, 1, 2])), shape=(1, 3))
    truth = coo_array(([1, 0, 0], ([0, 0, 0], [0, 1, 2])), shape=(1, 3))
    scores = coo_array(([0.6, 0.6, 0.2], ([0, 0, 0], [0, 1, 2])), shape=(1, 3))
    result = pair_counts(scores, truth, folds)
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 0)
    assert result["row_mean_half_tie_auc"] == 0.75
    assert result["row_mean_inverted_strict_auc"] == 0
