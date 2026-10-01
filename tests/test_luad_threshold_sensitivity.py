import numpy as np
import pandas as pd

from evals.luad_threshold_sensitivity import is_steroid, rank, summarize


def test_steroid_rule_on_known_names():
    assert is_steroid("hydrocortisone") and is_steroid("mometasone") and is_steroid("fluorometholone")
    assert not is_steroid("wortmannin") and not is_steroid("GDC-0941") and not is_steroid("floxuridine")


def test_rank_and_summary_overlap():
    genes = [f"G{i}" for i in range(6)]
    drugs = pd.DataFrame({"a": [-1, -1, -1, 1, 1, 1], "b": [1, 1, 1, -1, -1, -1], "c": [0.1] * 6},
                         index=genes, dtype=float)
    ranks = rank(drugs, np.array([1.0, -1.0, 0.0]), {"G0", "G1", "G2"}, {"G3", "G4", "G5"})
    assert ranks.index[0] == "a"
    s = summarize(ranks, ["a", "c"], ["b"])
    assert s["overlap_with_frozen"] == 2 and s["control_mean_percentile"] == 1.0
