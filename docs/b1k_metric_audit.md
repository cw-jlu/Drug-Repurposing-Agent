# B1k official NS-AUC diagnostic

Status: two **previously scored** frozen seeds were reproduced as diagnostics, one per splitter. The official CSVs and primary 100-run comparison are unchanged. Run `python -m evals.audit_b1k_ns_auc --split random_simple --seed-index 0 --output artifacts/reports/b1k_ns_auc_random_diagnostic.json` and likewise `--split weakly_correlated --output artifacts/reports/b1k_ns_auc_weak_diagnostic.json`. Each run emits a SHA-256-chained JSONL trace and verifies its three data-file hashes plus the frozen result and seed CSV hashes. The [random](../benchmark/results/b1k_ns_auc_random_seed0_diagnostic.json) and [weak](../benchmark/results/b1k_ns_auc_weak_seed0_diagnostic.json) diagnostic reports and their chained traces under `benchmark/results/traces/` are versioned.

The pinned runner calls `benchscofi.utils.rowwise_metrics.calc_auc` with `transpose=False`, so NS-AUC is an **unweighted mean over eligible drug rows**. Its comparison is strictly `positive_score > nonpositive_score`; an exact tie contributes zero, unlike standard AUC's half credit. Rows without both labels are omitted. The official global AUC and earlier project-level AUC use different aggregation/score handling and must not be used to infer a reversed sign.

| Frozen seed 17263407 | Saved = reproduced official NS-AUC | Eligible drug rows | Positive–nonpositive pairs: win / tie / loss | Row-mean AUC if ties got half credit | Row-mean AUC if score sign inverted, still strict |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random simple | 0.1466845878 | 60 | 360 / 2077 / 43 | 0.563887 | 0.018910 |
| Weakly correlated | 0.0588235294 | 34 | 300 / 10115 / 23 | 0.527157 | 0.004510 |

This establishes that **ties are the main observed reason** these two B1k NS-AUC values are extremely low; inverting the score would make them worse. The B1k neighbor-label propagation generates many identical scores in this sparse association matrix. The half-tie figures are counterfactual diagnostics, **not** new official benchmark scores and not a justification to revise the frozen comparator after viewing results. The audit does not show that every one of the 100 seeds has the same tie fraction, nor establish clinical prediction value. A prospective metric-definition change would require rerunning all methods and the 11 published baselines under the same new metric before a SOTA comparison.
