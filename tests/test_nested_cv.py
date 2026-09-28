import numpy as np
import pandas as pd

from benchmarks.recess_adapter.nested_cv import LocalDataset, random_simple_split


def test_local_nested_split_preserves_outer_boundary():
    ratings = pd.DataFrame(
        np.array([[1, 0, -1, 0], [0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1]]),
        index=["a", "b", "c", "d"], columns=["u1", "u2", "u3", "u4"])
    features = pd.DataFrame(np.eye(4), index=["g1", "g2", "g3", "g4"],
                            columns=ratings.index)
    users = pd.DataFrame(np.eye(4), index=features.index, columns=ratings.columns)
    dataset = LocalDataset(ratings, features, users)
    train, test = random_simple_split(dataset, 0.25, 1234)
    assert train.multiply(test).nnz == 0
    assert train.nnz + test.nnz == ratings.size
    assert np.isnan(dataset.subset(train).ratings.toarray()[test.row, test.col]).all()
