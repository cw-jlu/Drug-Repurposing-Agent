from evals.luad_positive_control_stats import statistics


def test_top_ranks_are_significant_and_middle_ranks_are_not():
    top = statistics({"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}, n_names=1000, draws=5000)
    assert top["permutation_p_one_sided"] < 0.01 and top["top10pct_count"] == 5
    mid = statistics({"a": 450, "b": 500, "c": 550}, n_names=1000, draws=5000)
    assert mid["permutation_p_one_sided"] > 0.3
