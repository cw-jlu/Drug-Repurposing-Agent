# B2 in the published RECeSS TRANSCRIPT runner

## Exact comparison target

This extension pins [benchmark-code](https://github.com/RECeSS-EU-Project/benchmark-code)
at commit `a7f11077271cedf3a98a82e3dc74b6fc0e93986e` and the authors'
[benchmark-results](https://github.com/RECeSS-EU-Project/benchmark-results)
at commit `cf5d9fdcb1ccd1676c7a0e7a39e784d79557fae0`. The comparison is
limited to TRANSCRIPT v2.0.0 and its 11 published reference models. B2 is
registered in the upstream pipeline by `benchmarks/recess_adapter/official_b2.patch`.
The patch also replaces Unix-only directory and cleanup commands with Python
calls, converts the generated NumPy seed to a Python integer, and restores an alias
required by the pinned `cute-ranking` package on NumPy 2. These changes do
not alter split, training, selection, or metric functions.
The run retained its small per-seed intermediate CSVs in the ignored artifact
directory; the versioned aggregate and seed CSVs are the comparison inputs.
The B2 adapter caches feature-only computations across folds; its scores are
tested for exact equality with the frozen B2 implementation, including after
validation labels are changed.

The upstream runner generates the same 100 seeds for each model with
`np.random.seed(1234)` followed by `np.random.choice(range(int(1e8)), size=100)`.
Each seed makes a 20% outer split with `random_simple_split` or
`weakly_correlated_split`. `cv_training` trains five inner-fold models using
`cv_type="random"` and chooses the one with the highest inner row-wise AUC.
For B2, `neighbors=10` and `rrf_k=60` are fixed before these runs. The
published `K=5` operation is fold-based model selection; it does not search a
hyperparameter grid. Our separate three-fold B2 grid study remains a distinct
experiment and must not be mixed into this table.
All 22 published TRANSCRIPT parameter JSON files (11 models × two split
types) set `params` to `null`, confirming that the reference result files do
not represent a five-fold hyperparameter grid search either.

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

## Result

B2 achieves NS-AUC `0.5222 ± 0.0354` on random simple (rank 9/12) and
`0.5019 ± 0.0036` on weakly correlated (rank 8/12). The leading published
references are BNNR (`0.7331`) and MBiRW (`0.7384`), respectively. B2 is
below both on all 100 paired seeds. The [full table](benchmark_results.md)
and [machine-readable comparisons](../benchmark/results/recess_official_b2_vs_11.json)
include all 11 references, paired differences, and source hashes. B2's
aggregate result, seed, and parameter CSVs are versioned in
`benchmark/results/recess_official_b2/`.

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
# To recompute only the comparison from the committed B2 CSVs:
python -m scripts.compare_recess_official_b2 --published $published --ours benchmark/results/recess_official_b2
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

As a cross-check of the local patched runner and TRANSCRIPT input, an
independent `LogisticMF` run on the first official random seed (`17263407`)
exactly reproduced the authors' published values for NS-AUC
(`0.6704512218407673`), global AUC (`0.950127861336678`), and global NDCG
(`0.5501272684100554`). This checks one reference run, not all 11 × 100 runs.

## B3: row-oriented fusion (2026-09-28)

The official NS-AUC ("Lin's AUC", `rowwise_metrics.calc_auc(..., transpose=False)`) ranks diseases **within each drug row** and counts tied scores as losses (strict `>`). B2 ranked candidates within disease columns, and its popularity component is constant within a drug row. B3 (`benchmarks/recess_adapter/official_b3.py`) rank-normalises six training-free components within drug rows and averages them: disease popularity, drug- and disease-expression kNN, label co-occurrence drug/disease CF, and reversal. It adds a deterministic disease-popularity tie-breaker.

The configuration was frozen in `configs/b3_row_fusion_v1.json` at commit `92943bb`, before any official scoring. It was selected on eight random-split development seeds disjoint from the 100 official seeds. `weakly_correlated_split` has no seed-dependent draw, so its development run necessarily used the official outer holdout; this is disclosed in the config and was not used for selection.

| Split | B3 NS-AUC (mean ± SD, 100 seeds) | Rank / 13 | Best published | B3 paired wins vs best |
|---|---:|---:|---|---:|
| Random simple | 0.7234 ± 0.0339 | 2 | BNNR 0.7331 | 35/100 |
| Weakly correlated | 0.6919 ± 0.0088 | 3 | MBiRW 0.7384 | 0/100 |

B3 does not surpass the best published model on either split. The gain from B2 (0.5222 → 0.7234) comes from matching the metric's orientation and from known-association structure, not from signature reversal. Machine-readable comparison: `benchmark/results/recess_official_b3_vs_11.json`; aggregate CSVs: `benchmark/results/recess_official_b3/`. The post-hoc variant B3noREV (reversal removed) completed afterwards: 0.7124 random simple (rank 2/13) and 0.7050 weakly correlated (rank 2/13, behind MBiRW). Because REV removal was chosen after the weak-split development score had been seen, B3noREV is reported only as a secondary, post-hoc result; B3 remains the primary pre-registered model. Ranks here are within {11 published models, B2, the model} (`rank_in_published_field`).
