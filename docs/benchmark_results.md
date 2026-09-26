# TRANSCRIPT benchmark results

Data: TRANSCRIPT v2.0.0 (613 drugs × 151 diseases; 401 positive, 11 explicit negative, 92,151 unknown associations). Source hashes are in [`data/manifests/transcript-v2.0.0.json`](../data/manifests/transcript-v2.0.0.json). The 100-run publication-protocol comparison below uses the authors' NS-AUC metric. The earlier project evaluations farther down use different seeds, selection rules and global AUC definitions; their numbers must not be compared directly with this table.

## B2 versus the 11 published RECeSS models: 100 runs, five folds

B2 was inserted into the pinned official runner, using the same TRANSCRIPT v2.0.0 input, 100 seed values in the same order, 20% outer holdout, five inner folds, row-wise AUC model selection, and `Lin's AUC` (the paper's NS-AUC) calculation as the 11 author-published reference models. Every metric has 100 finite values. Values are mean ± sample SD; each split has its own ranking.

| Method | Random simple NS-AUC | Weakly correlated NS-AUC |
| --- | ---: | ---: |
| BNNR | 0.7331 ± 0.0372 | 0.6061 ± 0.0242 |
| MBiRW | 0.6981 ± 0.0325 | 0.7384 ± 0.0137 |
| LogisticMF | 0.6704 ± 0.0483 | 0.6575 ± 0.0583 |
| SCPMF | 0.5991 ± 0.0503 | 0.4299 ± 0.0742 |
| HAN | 0.5979 ± 0.0509 | 0.7029 ± 0.0219 |
| PMF | 0.5746 ± 0.0354 | 0.5852 ± 0.0173 |
| FastaiCollabWrapper | 0.5516 ± 0.0573 | 0.6059 ± 0.0508 |
| ALSWR | 0.5511 ± 0.0399 | 0.6539 ± 0.0229 |
| **B2 (ours)** | **0.5222 ± 0.0354** | **0.5019 ± 0.0036** |
| NIMCGCN | 0.4778 ± 0.0325 | 0.4858 ± 0.0087 |
| DDA_SKF | 0.3300 ± 0.0438 | 0.5000 ± 0.0000 |
| LibMF | 0.2570 ± 0.0305 | 0.2247 ± 0.0000 |

B2 ranks 9/12 on random simple and 8/12 on weakly correlated. Against the strongest reference in each split, its paired mean NS-AUC difference is −0.2109 versus BNNR (95% paired bootstrap interval −0.2204 to −0.2013) and −0.2365 versus MBiRW (descriptive interval −0.2391 to −0.2338); B2 wins 0/100 paired seeds in both comparisons. The weakly correlated splitter repeats one outer holdout across its 100 seeds, so its interval reflects inner-fold/model-selection variation rather than 100 independent drug-group splits.

The authors' 22 parameter files use `params=null`; the official five-fold operation chooses the best fold-trained model but does not search hyperparameters. B2 uses its pre-fixed `neighbors=10`, `rrf_k=60`. Its official global AUC means (0.9446 random; 0.8967 weak) are substantially higher than its NS-AUC because the upstream global calculation scores the full matrix, including zero scores outside the validation fold. NS-AUC is the primary comparison here. One locally rerun LogisticMF seed exactly matched the published NS-AUC, global AUC and global NDCG. The other ten reference models use the authors' published CSVs rather than local reruns.

See the [full protocol and caveats](recess_official_comparison.md), [machine-readable 12-model comparison](../benchmark/results/recess_official_b2_vs_11.json), and the original B2 result/seed/parameter CSVs in `benchmark/results/recess_official_b2/`. This is a comparison with the paper's 11-model panel, not a claim against every later method.

## Frozen B2 component ablation and prescore LLM choice

The same official runner evaluated B2's three separate inputs under the same 100 seeds and five folds. The NS-AUC means ± sample SD are: B0p 0.5000 ± 0.0000 / 0.5000 ± 0.0000; B1k 0.1747 ± 0.0393 / 0.0562 ± 0.0466; B1 0.4816 ± 0.0340 / 0.5417 ± 0.0000; B2 0.5222 ± 0.0354 / 0.5019 ± 0.0036 (random simple / weakly correlated). Thus fusion helps relative to B1 on random simple but hurts it on the repeated weakly correlated outer holdout. A single DeepSeek call, committed before the component results were inspected, chose B2 for random and B1 for weakly correlated; these happen to be the best of these four methods on their respective splits. A new checksum-verified paired audit (`python -m evals.audit_historical_selector_v1`) confirms that the random choice ties fixed B2 on all 100 seeds and the weak-split B1 choice exceeds B2 by **0.039825 NS-AUC** on average (100/100 paired seed values higher; descriptive 5th–95th percentile of seed differences 0.0361–0.0471). Those 100 weak-split values share one held-out test set: they are **not** 100 independent demonstrations of benefit. Two choices are not a generalization test of LLM-based model selection, and the comparison remains below strong published models. See the [ablation and LLM evidence-triage protocol](component_and_llm_adjudication.md), [machine-readable paired audit](../benchmark/results/recess_component_selector_v1_paired_audit.json) with its JSONL trace receipt, and [component results](../benchmark/results/recess_official_component_ablation.json). The separate LUAD LLM triage prioritized research actions, not clinical treatments or benchmark scores.

## Earlier project evaluations

The tables below use means ± sample SD over seeds 1234, 1235 and 1236 with `test_size=0.2`. The fixed methods are defined in [the protocol](evaluation_protocol.md).

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

The weakly correlated split produced identical scores for all deterministic methods and identical validation label counts across these three seeds. Inspection of the `stanscofi` 2.0.1 splitter confirms that, for fixed item features, its clustering and outer-fold assignment use no random draw after setting the seed; the same outer holdout is repeated. The apparent zero SD does not imply low uncertainty. B0 changes because its random scores use the seed. RRF assigns tied component scores equal average ranks, removing an implicit input-row-order tie break. The saved B2 runs and this table were regenerated after that correction.

## Nested tuning of B1k and B2

The new nested runner uses the same three outer seeds and both outer split protocols. Hyperparameters are selected only by three-fold CV inside each outer training set; leakage audits in all six JSON files report zero outer-test coordinates seen by inner CV.

| Outer split | Method | Selected parameters | Outer global AUC | Outer global NDCG |
| --- | --- | --- | ---: | ---: |
| Random simple | B1k | neighbors=40 in 3/3 runs | 0.6206 ± 0.0060 | 0.4168 ± 0.0122 |
| Random simple | B2 | neighbors=20, RRF k=20 in 3/3 | 0.7371 ± 0.0384 | 0.4450 ± 0.0609 |
| Weakly correlated | B1k | neighbors=40 in 3/3 | 0.5140 ± 0.0000 | 0.3768 ± 0.0000 |
| Weakly correlated | B2 | neighbors=20 once / 10 twice; RRF k=20 | 0.4702 ± 0.0079 | 0.3421 ± 0.0136 |

Tuning materially improves B1k over its fixed 10-neighbor version on random splits (AUC 0.5660 → 0.6206), but B2 improves only slightly there (0.7270 → 0.7371) and does not improve under the weakly correlated outer split (0.4752 → 0.4702). Thus this project-level procedure strengthens the negative conclusion: expression-neighbor tuning does not solve generalization to dissimilar compounds. LogisticMF was the strongest of the three direct package baselines tested at this stage. The saved files are `benchmark/results/nested_cv_*.json`.

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

These earlier project evaluations remain **limited external evaluations**. Their grids are intentionally bounded and their values are not the 100-run publication-protocol scores above. The PMF runner restores the removed NumPy `np.int` alias. Unknown `0` entries are scored as non-positive under the global AUC convention, not established clinical failures.

Reproduce one run with:

```powershell
python benchmarks/recess_adapter/run.py --data data/raw/TRANSCRIPT_dataset_v2.0.0 --split random_simple --seed 1234
python benchmarks/recess_adapter/nested_cv.py --data data/raw/TRANSCRIPT_dataset_v2.0.0 --split random_simple --seed 1234
python -m benchmarks.recess_adapter.nested_cv_official --data data/raw/TRANSCRIPT_dataset_v2.0.0 --split random_simple --seed 1234
```
