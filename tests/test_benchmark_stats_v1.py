import numpy as np

from evals.benchmark_stats_v1 import nadeau_bengio, paired


def test_correction_widens_interval_relative_to_naive():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.02, 100)
    nb = nadeau_bengio(d)
    naive_half = 1.984 * d.std(ddof=1) / np.sqrt(100)
    assert (nb["mean_ci95"][1] - nb["mean_ci95"][0]) / 2 > 4 * naive_half


def test_zero_mean_is_not_significant():
    d = np.array([0.01, -0.01] * 50)
    assert nadeau_bengio(d)["p_two_sided"] > 0.9


def test_weak_split_reports_descriptive_only():
    out = paired(np.ones(5), np.zeros(5), "weakly_correlated")
    assert out["wins"] == 5 and "nadeau_bengio" not in out and "note" in out
