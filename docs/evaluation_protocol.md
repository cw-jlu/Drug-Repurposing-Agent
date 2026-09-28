# TRANSCRIPT evaluation protocol

The external reference is [RECeSS benchmark-code](https://github.com/RECeSS-EU-Project/benchmark-code), using `stanscofi` 2.0.1. Our five methods are fixed before looking at official validation results:

| ID | Method | Reads training labels? |
| --- | --- | --- |
| B0 | Seeded random score | No |
| B0p | Per-drug positive count in training fold | Yes |
| B1 | Negative Spearman correlation of expression vectors | No |
| B1k | Drug-expression 10-neighbor positive-label propagation | Yes |
| B2 | Fixed RRF of B0p, B1k, B1; `k=60` | Yes |

`stanscofi.Dataset` marks every non-NaN rating as available in `folds`, including `0` (unknown). Its `random_simple_split` stratifies available ratings by their actual `-1`, `0`, `1` values. `dataset.subset(train_folds)` masks held-out entries to NaN. The adapter's `fit` reads only that training subset. `predict_proba` may receive a validation `Dataset` object that internally contains labels, but only reads `items`, `users`, IDs, and `folds`; it does not access `ratings`.

In the official benchmark pipeline, global AUC converts all validation ratings less than 1 to 0, so unknown `0` and explicit failed `-1` are both counted in its non-positive class. That is the **official scoring convention**, not a biological claim that unknown pairs are ineffective. Official global accuracy separately excludes unknown ratings (`y != 0`). Row-wise metrics may be undefined for disease rows without both classes; report coverage. We preserve the supplied split and seed, and save them with every result.

The runner also computes supplemental disease-wise score AUC (only diseases with both classes in their test fold), mean Recall@10 (only diseases with at least one test positive), and mean reciprocal rank under the same official convention for unknown pairs. These are clearly labeled as project-defined supplements; the two headline metrics call `stanscofi.validation.AUC` and `NDCGk` directly.

`benchmarks/recess_adapter/nested_cv.py` adds a three-fold inner selection loop inside each untouched outer training fold. It tunes B1k neighbors over 5/10/20/40 and B2 over neighbors 5/10/20 plus RRF `k` 20/60/100, selecting by mean inner global AUC with NDCG and sorted parameters as deterministic tie-breakers. Every result records that outer-train/test overlap and outer-test coordinates observed by inner CV are both zero. The lightweight runner reproduces the stanscofi 2.0.1 split code and its sklearn AUC / cute-ranking NDCG definitions without importing optional plotting and deep-learning dependencies. This is a project nested-CV protocol for B1k/B2, not a reproduction of every RECeSS publication model or search space.

The CLI's label-free expression ranking reads only `items.csv` and `users.csv`; it never needs `ratings_mat.csv`. The benchmark runner may read ratings in its split/evaluation process, with no label path supplied to the ranking workflow.

The additional ALSWR, PMF, and LogisticMF study uses the official `benchscofi` 2.0.1 model classes and `predict_proba` paths. For each of `random_simple` and `weakly_correlated`, it runs outer seeds 1234–1238 and selects among four prespecified parameter configurations with three-fold inner CV. Selection uses mean inner global AUC with NDCG and sorted parameters as deterministic tie-breakers. Training subsets explicitly zero unavailable ratings; prediction datasets contain zero ratings; each result records zero overlap between outer train/test coordinates and zero exposure of outer-test coordinates to inner CV. This is a project search grid, not a claim to reproduce every hyperparameter in the RECeSS publication.

The separate [publication-protocol comparison](recess_official_comparison.md) puts fixed B2 into the authors' pinned runner and compares it with all 11 author-published TRANSCRIPT models using the same 100 seed values, `K=5`, 20% outer split and NS-AUC implementation. The authors' `K=5` selects the highest inner row-wise AUC fold-trained model; the published parameter files have `params=null`, so this is not a five-fold hyperparameter grid search. The weakly correlated splitter repeats one outer holdout across seeds. Its results and the earlier project global-AUC tables use different evaluation denominators and should be interpreted separately.
