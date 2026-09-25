# Preliminary TRANSCRIPT benchmark results

Data: TRANSCRIPT v2.0.0 (613 drugs × 151 diseases; 401 positive, 11 explicit negative, 92,151 unknown associations). Source hashes are in [`data/manifests/transcript-v2.0.0.json`](../data/manifests/transcript-v2.0.0.json). All rows use the same `stanscofi` 2.0.1 splitting and global AUC/NDCG functions. The fixed methods are defined in [the protocol](evaluation_protocol.md). Scores below are means ± sample SD over seeds 1234, 1235, and 1236; `test_size=0.2`.

## Random simple split

| Method | Global AUC | Global NDCG |
| --- | ---: | ---: |
| B0 seeded random | 0.5363 ± 0.0061 | 0.3425 ± 0.0016 |
| B0p train-fold drug popularity | 0.7346 ± 0.0453 | 0.4279 ± 0.0088 |
| B1 negative Spearman | 0.4820 ± 0.0108 | 0.3384 ± 0.0036 |
| B1k drug-expression kNN | 0.5660 ± 0.0102 | 0.4051 ± 0.0156 |
| B2 fixed RRF of B0p/B1k/B1 | 0.7270 ± 0.0417 | 0.4543 ± 0.0530 |
| ALSWR (benchscofi defaults) | 0.6253 ± 0.0110 | 0.4421 ± 0.0142 |
| PMF (benchscofi defaults) | 0.6075 ± 0.0242 | 0.3583 ± 0.0110 |
| LogisticMF (benchscofi defaults) | 0.8005 ± 0.0308 | 0.4863 ± 0.0284 |

Supplemental disease-wise score AUC was calculable for 57, 50, and 51 disease rows across the three test folds, respectively. Their mean AUCs were B0 0.5377, B0p 0.7233, B1 0.4734, B1k 0.5381, and B2 0.7045. This metric is project-defined and uses continuous scores; it is separate from the reference pipeline's thresholded row-wise metric.

## Weakly correlated split

| Method | Global AUC | Global NDCG |
| --- | ---: | ---: |
| B0 seeded random | 0.5049 ± 0.0419 | 0.3288 ± 0.0071 |
| B0p train-fold drug popularity | 0.5000 ± 0.0000 | 0.3287 ± 0.0000 |
| B1 negative Spearman | 0.5238 ± 0.0000 | 0.3412 ± 0.0000 |
| B1k drug-expression kNN | 0.5108 ± 0.0000 | 0.3441 ± 0.0000 |
| B2 fixed RRF of B0p/B1k/B1 | 0.4752 ± 0.0000 | 0.3448 ± 0.0000 |
| ALSWR (benchscofi defaults) | 0.6481 ± 0.0000 | 0.3550 ± 0.0000 |
| PMF (benchscofi defaults) | 0.5656 ± 0.0031 | 0.3388 ± 0.0019 |
| LogisticMF (benchscofi defaults) | 0.6696 ± 0.0495 | 0.3835 ± 0.0236 |

The weakly correlated split produced identical scores for all deterministic methods and identical validation label counts across these three seeds. This suggests the selected fold may have been the same; the fold coordinates have not been independently compared. The apparent zero SD does not imply low uncertainty. B0 changes because its random scores use the seed. RRF now assigns tied component scores equal average ranks, removing an implicit input-row-order tie break. The saved B2 runs and this table were regenerated after that correction.

## Nested tuning of B1k and B2

The new nested runner uses the same three outer seeds and both outer split protocols. Hyperparameters are selected only by three-fold CV inside each outer training set; leakage audits in all six JSON files report zero outer-test coordinates seen by inner CV.

| Outer split | Method | Selected parameters | Outer global AUC | Outer global NDCG |
| --- | --- | --- | ---: | ---: |
| Random simple | B1k | neighbors=40 in 3/3 runs | 0.6206 ± 0.0060 | 0.4168 ± 0.0122 |
| Random simple | B2 | neighbors=20, RRF k=20 in 3/3 | 0.7371 ± 0.0384 | 0.4450 ± 0.0609 |
| Weakly correlated | B1k | neighbors=40 in 3/3 | 0.5140 ± 0.0000 | 0.3768 ± 0.0000 |
| Weakly correlated | B2 | neighbors=20 once / 10 twice; RRF k=20 | 0.4702 ± 0.0079 | 0.3421 ± 0.0136 |

Tuning materially improves B1k over its fixed 10-neighbor version on random splits (AUC 0.5660 → 0.6206), but B2 improves only slightly there (0.7270 → 0.7371) and does not improve under the weakly correlated outer split (0.4752 → 0.4702). Thus the more complete procedure strengthens the negative conclusion: expression-neighbor tuning does not solve generalization to dissimilar compounds, and LogisticMF remains the strongest tested baseline. The saved files are `benchmark/results/nested_cv_*.json`.

## Nested tuning of ALSWR, PMF, and LogisticMF

The three official `benchscofi` 2.0.1 models now use three-fold inner selection inside five outer seeds (1234-1238). Each prediction call receives a feature-identical test object whose rating matrix contains zeros only. All ten saved runs report zero outer train/test overlap and zero outer-test coordinates in inner CV.

| Outer split | Method | Outer global AUC | Outer global NDCG | Most frequent selected parameters |
| --- | --- | ---: | ---: | --- |
| Random simple | ALSWR | 0.6579 ± 0.0320 | 0.4908 ± 0.0518 | 20 factors, alpha 5, reg 0.1 in 3/5 |
| Random simple | PMF | 0.6161 ± 0.0211 | 0.3600 ± 0.0084 | 15 factors, 160 iterations, lr 0.1, reg 0.01 in 5/5 |
| Random simple | LogisticMF | 0.7826 ± 0.0410 | 0.4811 ± 0.0229 | 2 factors, reg 0.6 in 5/5 |
| Weakly correlated | ALSWR | 0.5922 ± 0.0528 | 0.3628 ± 0.0215 | no single configuration exceeded 2/5 |
| Weakly correlated | PMF | 0.5654 ± 0.0036 | 0.3394 ± 0.0029 | 15 factors, 160 iterations, lr 0.1, reg 0.01 in 5/5 |
| Weakly correlated | LogisticMF | 0.6474 ± 0.0678 | 0.3817 ± 0.0205 | four configurations selected across five seeds |

Nested tuning leaves LogisticMF as the strongest of these three models, but its weakly correlated result varies substantially by initialization and inner split. The five-seed mean is lower than the earlier three-seed default estimate in both protocols. This is a more conservative and better isolated result, not a publication reproduction.

These remain **limited external evaluations**, not the full RECeSS publication protocol. The grids are intentionally bounded, no SOTA claim is supported, and the values are not quoted publication scores. The PMF runner restores the removed NumPy `np.int` alias. Unknown `0` entries are scored as non-positive under the official global AUC definition, not established clinical failures.

Reproduce one run with:

```powershell
python benchmarks/recess_adapter/run.py --data data/raw/TRANSCRIPT_dataset_v2.0.0 --split random_simple --seed 1234
python benchmarks/recess_adapter/nested_cv.py --data data/raw/TRANSCRIPT_dataset_v2.0.0 --split random_simple --seed 1234
python -m benchmarks.recess_adapter.nested_cv_official --data data/raw/TRANSCRIPT_dataset_v2.0.0 --split random_simple --seed 1234
```
