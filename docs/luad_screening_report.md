# LUAD transcriptomic screening report

This is a reproducible *in vitro* expression screen, not a treatment recommendation. The ranked output is in `artifacts/reports/luad_eh3226/all_candidates.csv` after running `scripts/rank_luad_eh3226.py`; input and output hashes are in [`data/manifests/eh3226-luad.json`](../data/manifests/eh3226-luad.json). The 2.46 GB EH3226 input is fetched separately with `scripts/fetch_eh3226.py`.

## Prespecified method and data

GSE32863 contributes 57 verifiable LUAD tumor/normal pairs. The paired exploratory t-test creates the disease log2 fold-change vector. ExperimentHub EH3226 contributes A549 drug perturbation signatures at 10 µM for 24 hours. On 961 aligned landmark genes, the ranking combines negative Spearman correlation and an up/down gene-set connectivity score using reciprocal-rank fusion (`k=60`). In the aligned landmark set, 57 genes meet the predetermined up threshold and 50 meet the down threshold. No positive-control label or drug-target annotation enters the score.

## Top-10 hypotheses

| Rank | Drug name in EH3226 | Broad sample identity audit |
| ---: | --- | --- |
| 1 | hydrocortisone | Same connectivity; different stereochemistry |
| 2 | beclomethasone-dipropionate | Exact InChIKey |
| 3 | clocortolone-pivalate | Exact InChIKey |
| 4 | mometasone | Identity mismatch |
| 5 | flumetasone | Ambiguous GEO name |
| 6 | BRD-K84203638 | No Hub sample |
| 7 | hydrocortisone-hemisuccinate | Same connectivity; different stereochemistry |
| 8 | fluorometholone | No Hub sample |
| 9 | fluticasone | Ambiguous GEO name |
| 10 | diflorasone | No Hub sample |

The [Broad Drug Repurposing Hub](https://repo-hub.broadinstitute.org/repurposing) annotates several of these names as steroids or glucocorticoid-receptor agonists. The strong class concentration may reflect a common A549 transcriptional response; it does not establish therapeutic benefit in LUAD. The two exact InChIKey matches refer to inferred GEO perturbagen IDs. EH3226 itself lacks per-column IDs, so even these matches do not prove the precise sample used upstream. Name-level targets, clinical phases, and indications were not promoted to candidate evidence ledgers.

## Frozen-reference check

Eleven reference names were fixed before ranking. Five exist in the 4,920-name EH3226 A549 subset: docetaxel ranks 106, crizotinib 241, gefitinib 2,444, paclitaxel 2,962, and erlotinib 3,147. The other six are absent and are **unmeasured**, not failures. The spread of these ranks and the top-ranked steroid cluster mean this screen has not demonstrated LUAD drug recovery. No AUC was calculated by treating unmeasured or unknown drugs as negatives.

## Evidence and limitations

All ten candidates retain `insufficient_evidence` in their structured ledgers. A frozen [NCBI PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25499/) title/abstract search for each exact drug name with LUAD/NSCLC terms returned 39 hits for hydrocortisone, one for beclomethasone dipropionate, and zero for the other eight names. This narrow query is a retrieval snapshot, not an evidence-absence test; synonyms and broader terms were not searched. Two records were checked at the abstract level and placed in `context_evidence` only: [PMID 35676421](https://pubmed.ncbi.nlm.nih.gov/35676421/) independently proposes hydrocortisone as a computational NSCLC candidate, without efficacy validation in its abstract; [PMID 12538830](https://pubmed.ncbi.nlm.nih.gov/12538830/) reports glucocorticoid-receptor-dependent CYP3A5 induction by beclomethasone dipropionate in A549 cells, a mechanistic observation rather than anticancer efficacy. Other retrieved hits remain unreviewed.

No candidate has been independently validated for LUAD efficacy, dose relevance, safety, or patient context. The 10 µM single-cell-line condition, first-duplicate selection in EH3226, and uncertain drug IDs are material limitations. The paired limma sensitivity analysis retained the same thresholded disease gene sets, but compound-level reconciliation, full literature review with supporting and opposing sources, and clinical-context assessment are still needed before an evidence-ranked Top-10 can be issued. The current expression rank is useful for prioritizing research only.
