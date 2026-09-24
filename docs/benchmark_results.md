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

These are **initial external evaluations**, not the full RECeSS publication protocol: the runners use the official split and global metric functions, but there is no nested hyperparameter CV and only three seeds. ALSWR, PMF, and LogisticMF are direct runs of `benchscofi` 2.0.1 with package defaults; they are not quoted publication scores. The PMF runner restores the removed NumPy `np.int` alias, and all three models receive a label-neutral test object during prediction. The numerical results are saved in `benchmark/results/recess_defaults_*.json`. No SOTA claim is supported. LogisticMF exceeds B2 in both splits on these runs. B1 is near chance; adding it to a label-driven fusion did not outperform B0p on random splits and performed worse than B0p on weakly correlated splits. Unknown `0` entries are scored as non-positive under the official global AUC definition, not established clinical failures.

Reproduce one run with:

```powershell
python benchmarks/recess_adapter/run.py --data data/raw/TRANSCRIPT_dataset_v2.0.0 --split random_simple --seed 1234
```
