# Current limitations

- A reversal score measures expression opposition, not patient benefit or drug safety.
- The TRANSCRIPT pilot found reversal near chance. B2's added reversal term may reduce performance; it is retained as a preregistered comparison, not tuned on validation results.
- One TRANSCRIPT disease expression vector is constant. Its Spearman score is set to zero and logged by QC.
- Missing genes are handled by intersection for Spearman, but the connectivity method uses zero-imputed ranks. Interpret results with missing data cautiously.
- Pathway antagonism requires a separately versioned pathway gene set and is not used in this release.
- The LUAD GSE32863 disease signature and a derived A549 expression subset are available. EH3226 retains drug names rather than per-column IDs, but replaying its documented first-duplicate selection recovers one source GSE92742 `sig_id` and `pert_id` for each Top-10 column. Only two match Broad sample annotations by exact InChIKey; reconstruction does not prove cross-source chemical equivalence or clinical efficacy.
- Paired limma recovered the same thresholded LUAD gene sets as the t-test, but these observational disease-expression differences still do not prove a causal disease mechanism. The EH3226 subset keeps the first technical duplicate for a drug name/cell combination and does not cover all 5,814 GEO metadata signatures.
- B1k/B2 have three-fold inner tuning across three outer seeds and both split protocols. ALSWR, PMF, and LogisticMF have a bounded four-candidate grid, three-fold inner tuning, and five outer seeds for both split protocols; this is broader than a default rerun but not the publication's complete search space, and no SOTA claim is supported. The Top-10 literature matrix is a bounded human review, while the dose/subtype/combination document is a planned wet-lab protocol only. Jev access remains unavailable.
