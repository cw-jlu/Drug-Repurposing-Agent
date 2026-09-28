import pandas as pd
import pytest

from scripts.stage_method_selection_partition_v2_positive_only import positive_only_ratings


def test_positive_only_mapping_changes_only_explicit_negatives():
    original = pd.DataFrame([[-1, 0, 1], [1, -1, 0]], index=["d1", "d2"])
    amended, negatives = positive_only_ratings(original)
    assert negatives == 2
    assert amended.equals(pd.DataFrame([[0, 0, 1], [1, 0, 0]], index=["d1", "d2"]))
    assert original.iloc[0, 0] == -1


def test_positive_only_mapping_rejects_other_ratings():
    with pytest.raises(ValueError, match="Unexpected"):
        positive_only_ratings(pd.DataFrame([[2]]))
