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

The runner also computes supplemental disease-wise score AUC (only diseases with both classes in their test fold), mean Recall@10 (only diseases with at least one test positive), and mean reciprocal rank under the same official convention for unknown pairs. These are clearly labeled as project-defined supplements; the two headline metrics call `stanscofi.validation.AUC` and `NDCGk` directly. Fixed baselines need no hyperparameter search, so the runner does not call the reference nested-CV optimizer.

The CLI's label-free expression ranking reads only `items.csv` and `users.csv`; it never needs `ratings_mat.csv`. The benchmark runner may read ratings in its split/evaluation process, with no label path supplied to the ranking workflow.

The additional ALSWR, PMF, and LogisticMF runs use `benchscofi` 2.0.1 defaults on the same saved split settings. Their prediction object has zeroed test ratings so upstream model preprocessing cannot inspect held-out labels; evaluation reads labels separately from the test fold. These are direct baseline reruns without the publication's nested tuning protocol.
