# B2 in the published RECeSS TRANSCRIPT runner

## Exact comparison target

This extension pins [benchmark-code](https://github.com/RECeSS-EU-Project/benchmark-code)
at commit `a7f11077271cedf3a98a82e3dc74b6fc0e93986e` and the authors'
[benchmark-results](https://github.com/RECeSS-EU-Project/benchmark-results)
at commit `cf5d9fdcb1ccd1676c7a0e7a39e784d79557fae0`. The comparison is
limited to TRANSCRIPT v2.0.0 and its 11 published reference models. B2 is
registered in the upstream pipeline by `benchmarks/recess_adapter/official_b2.patch`.
The patch also makes directory creation and intermediate-file cleanup portable,
converts the generated NumPy seed to a Python integer, and restores an alias
required by the pinned `cute-ranking` package on NumPy 2. These changes do
not alter split, training, selection, or metric functions.

The upstream runner generates the same 100 seeds for each model with
`np.random.seed(1234)` followed by `np.random.choice(range(int(1e8)), size=100)`.
Each seed makes a 20% outer split with `random_simple_split` or
`weakly_correlated_split`. `cv_training` trains five inner-fold models using
`cv_type="random"` and chooses the one with the highest inner row-wise AUC.
For B2, `neighbors=10` and `rrf_k=60` are fixed before these runs. The
published `K=5` operation is fold-based model selection; it does not search a
hyperparameter grid. Our separate three-fold B2 grid study remains a distinct
experiment and must not be mixed into this table.

In `stanscofi` 2.0.1, `weakly_correlated_split` seeds random-number generators
but its clustering and fold assignment contain no subsequent random draw.
Consequently its 100 seeds repeat the same outer holdout; only the inner
five-fold split can change with the seed. This is part of the published
protocol, and the weak-split spread must not be interpreted as uncertainty
over 100 distinct drug-group holdouts.

The primary result is the upstream `rowwise_metrics.calc_auc` output labeled
`Lin's AUC` in the CSV and `NS AUC` in the authors' analysis. The same runner
also writes global AUC, global NDCG, row-wise AUC/NDCG and other metrics.
The summary script requires exact agreement of all 100 seed positions for B2
and every published model before producing any paired difference. It records
source-file SHA-256 values and finite-value coverage for each metric.

## Reproduce

From the repository root in PowerShell:

```powershell
$upstream = Join-Path $env:TEMP 'recess-benchmark-code-pinned'
$published = Join-Path $env:TEMP 'recess-benchmark-results-pinned'
git clone https://github.com/RECeSS-EU-Project/benchmark-code.git $upstream
git -C $upstream checkout a7f11077271cedf3a98a82e3dc74b6fc0e93986e
$patch = (Resolve-Path benchmarks/recess_adapter/official_b2.patch).Path
git -C $upstream apply --recount $patch
git clone --filter=blob:none --sparse https://github.com/RECeSS-EU-Project/benchmark-results.git $published
git -C $published checkout cf5d9fdcb1ccd1676c7a0e7a39e784d79557fae0
git -C $published sparse-checkout set results_TRANSCRIPT results_TRANSCRIPT_weakly_correlated
python -m scripts.run_recess_official_b2 --upstream $upstream --n 100 --k 5
python -m scripts.compare_recess_official_b2 --published $published
```

Install the environment dependencies from `requirements-benchmark.lock` and
`statsmodels` before running. The runner stages three checksum-verified input
CSV files from `data/raw/TRANSCRIPT_dataset_v2.0.0` using hard links when
possible. Its default outputs are under ignored `artifacts/recess_official_b2`;
the comparison summary is `benchmark/results/recess_official_b2_vs_11.json`.
For a smoke test, use `--n 1 --splitting random_simple` and a separate
`--output` directory. The upstream runner reuses a finished result CSV, so
do not reuse a smoke-test directory for the 100-run comparison.

## Interpretation boundary

Matching seeds, folds, source data version, and metric code makes this a
protocol-matched comparison against the 11 author-published result files.
The 11 models are not rerun in the local environment. Their original raw
dataset bytes and software builds are not independently checksum verified by
the published CSVs, so any environment or source-data discrepancy remains a
reproducibility limitation. The primary metric is a ranking measure for known
and unknown associations; it is not a clinical benefit estimate.
