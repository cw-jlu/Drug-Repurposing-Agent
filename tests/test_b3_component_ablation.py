from evals.b3_component_ablation import dev_seeds, variants


def test_variants_cover_full_leave_one_out_and_singletons():
    v = variants()
    assert len(v) == 13 and len(v["full"]) == 6
    assert all(len(v[f"-{c}"]) == 5 and v[f"only {c}"] == (c,) for c in v["full"])


def test_dev_seeds_exclude_official_seeds():
    import numpy as np
    official = set(np.random.RandomState(1234).choice(range(int(1e8)), size=100).tolist())
    seeds = dev_seeds(10)
    assert len(seeds) == 10 and not official & set(seeds)
