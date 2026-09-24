# Data card

## Purpose and scope

These public datasets support research on transcriptomic drug ranking and a lung adenocarcinoma (LUAD) case study. They are not a source of patient-level treatment labels or clinical benefit estimates. Raw and processed expression files are kept outside Git; source locations, byte counts, checksums, selection rules, and output checksums are recorded in `data/manifests/`.

## Sources

| Dataset | Role | Frozen processing and coverage | Provenance |
| --- | --- | --- | --- |
| [TRANSCRIPT v2.0.0](https://zenodo.org/records/7982976) | External benchmark | 613 drugs × 151 diseases; 401 known positive, 11 explicit negative, 92,151 unknown associations | `data/manifests/transcript-v2.0.0.json` |
| [GSE32863/GPL6884](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32863) | LUAD disease signature | 116 samples; 57 verifiable tumor/normal patient pairs after excluding two unmatched samples; 19,404 selected genes | `data/manifests/gse32863.json` |
| [GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742) metadata | Prespecified A549 cohort and drug IDs | 24 h, 10 µM, chemical treatments; 5,814 signatures / 5,169 IDs in metadata | `data/manifests/gse92742-a549.json` |
| [ExperimentHub EH3226](https://bioconductor.org/packages/release/data/experiment/html/signatureSearchData.html) | A549 Level 5 expression | Derived 10 µM/24 h subset; 4,920 A549 drug names after upstream first-duplicate selection; 961 aligned landmark genes | `data/manifests/eh3226-luad.json` |
| [Broad Drug Repurposing Hub](https://repo-hub.broadinstitute.org/repurposing) | Name and chemical-identity audit | Drug and sample annotations dated 2025-08-18 | `data/manifests/repurposing-hub-2025-08-18.json` |

## Processing choices

GSE32863 uses tumor-minus-normal log2 fold change from a paired t-test, Benjamini-Hochberg FDR adjustment, and predetermined FDR < 0.05 / absolute log2FC ≥ 1 thresholds. A paired limma sensitivity analysis with patient fixed effects finds the same thresholded up/down sets; see `data/manifests/gse32863-limma.json`. The LUAD drug screen uses negative Spearman and up/down gene-set connectivity fused by RRF with `k=60`; no benchmark labels enter this screen. Eleven reference treatment names were frozen before ranking. Five are measurable in EH3226 and six are absent.

## Coverage and quality limits

EH3226 stores drug names, not per-signature perturbagen IDs. In the matched GEO metadata, 206 of its 4,920 A549 names map to multiple IDs. The Top-10 Broad identity audit finds two exact InChIKey matches, two same-connectivity/different-stereochemistry cases, one mismatch, two ambiguous GEO names, and three names with no Hub sample. Name-level annotations must not be treated as chemical-identity proof. Single A549 cell-line conditions, 10 µM concentration, 24-hour exposure, first-duplicate selection, and missing clinical endpoints limit generalization.

## Intended and excluded uses

Intended: method development, transparent benchmarking, hypothesis generation, and research prioritization. Excluded: diagnosis, prescribing, clinical efficacy or safety claims, and treating unknown benchmark associations as proven failures. The Broad annotation files state a non-commercial-use restriction and a research-only disclaimer; downstream reuse must respect the source terms.
