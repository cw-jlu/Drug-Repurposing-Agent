# Current limitations

- A reversal score measures expression opposition, not patient benefit or drug safety.
- The TRANSCRIPT pilot found reversal near chance. B2's added reversal term may reduce performance; it is retained as a preregistered comparison, not tuned on validation results.
- One TRANSCRIPT disease expression vector is constant. Its Spearman score is set to zero and logged by QC.
- Missing genes are handled by intersection for Spearman, but the connectivity method uses zero-imputed ranks. Interpret results with missing data cautiously.
- Pathway antagonism requires a separately versioned pathway gene set and is not used in this release.
- The LUAD GSE32863 disease signature and a derived A549 expression subset are available. EH3226 retains drug names but not per-column perturbagen IDs; 206 names map to multiple GEO IDs, and only two of the Top-10 inferred IDs match Broad sample annotations by exact InChIKey. The resulting screen cannot establish clinical efficacy.
- Paired limma recovered the same thresholded LUAD gene sets as the t-test, but these observational disease-expression differences still do not prove a causal disease mechanism. The EH3226 subset keeps the first technical duplicate for a drug name/cell combination and does not cover all 5,814 GEO metadata signatures.
- Three `benchscofi` default baselines have been run, but nested hyperparameter CV, broader seeds, clinical evidence curation, Jev access, and course deliverables remain outstanding. The evidence ledger validates citation syntax; it does not prove a citation supports a claim.
