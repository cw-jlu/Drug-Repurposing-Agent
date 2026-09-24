# Project status (2026-09-24)

## Implemented and verified

- Label-free TRANSCRIPT expression loader, gene-ID alignment, QC, negative Spearman, rank connectivity, and fixed reciprocal-rank fusion.
- Controlled natural-language routing with provider-neutral tool schemas, a deterministic fallback planner, a structured external-LLM adapter boundary, mode-specific allow lists, missing-input states, and an auditable run trace.
- Live DeepSeek `deepseek-flash` function calling through an environment-only credential path. A frozen 20-case routing eval was run three times: 0.95, 0.85, and 1.00 accuracy (mean 0.933, sample SD 0.076), versus 0.75 for the deterministic rule planner. All returned calls passed local tool and argument validation.
- One live strict-mode TRANSCRIPT Agent run completed end to end: 12,096 genes, 613 drugs, 151 diseases, four score matrices, and a five-event Agent trace.
- Strict-mode command-line workflow with input SHA-256, output matrices, QC, trace, and limitations.
- RECeSS/stanscofi-compatible B0, B0p, B1, B1k, B2 adapter with training-fold label isolation.
- Three-seed random and weakly correlated external split runs using official global AUC/NDCG functions; result JSON and summary are committed.
- TRANSCRIPT v2.0.0 source checksum manifest and reproducible downloader.
- GSE32863/GPL6884 paired disease-signature processing, with 57 verified pairs and two unpaired samples excluded.
- A paired limma sensitivity analysis in R 4.6.1 / limma 3.68.5 reproduces the same 512 up and 749 down gene sets at the prespecified thresholds.
- GSE92742 A549 compound metadata cohort and prespecified reference names, frozen before any LUAD drug ranking.
- Three direct `benchscofi` default-baseline runs (ALSWR, PMF, LogisticMF) on each of the six saved RECeSS folds. LogisticMF leads the project's fixed methods in both split protocols; no SOTA claim is made.
- ExperimentHub EH3226 A549 Level 5 subset verified by SHA-256, with 4,920 drug-name signatures and 961 aligned landmark genes. A transcriptomic LUAD Top-10 and frozen-control ranks have been generated.
- Broad Repurposing Hub identity audit for the Top-10. Only two names have exact InChIKey matches between the inferred GEO ID and Hub sample annotation; several others are ambiguous or mismatched.
- A LUAD case packager validates frozen source/output hashes, ranks, ledgers, identity audit, and preregistered controls, then writes a trace and external-model cost record. It stops at `expression_screen_complete_evidence_pending`.
- Data card, system card, and limitations document.
- A pinned Python 3.10 dependency set in `requirements-benchmark.lock` and Windows setup script.
- An optional Jev Choice/Score/Noul HTTP adapter with response validation and fail-closed confidence gating; live Jev use has not been run without credentials.
- Twenty-five passing internal tests covering scientific signs, tied ranks, missing/constant vectors, label isolation, provenance, pairing, citation syntax, PubMed triage, Jev response gating, natural-language routing, missing-input handling, tool allow-list enforcement, and DeepSeek response parsing.
- An editable four-page course-report draft rendered and visually checked page by page, plus a nine-slide editable defense deck with checked slide previews and native charts. Both include the controlled Agent flow and reflect the current evidence limits.

## Needed for the planned final deliverable

1. Independently review targets, clinical context, supporting and contradicting literature for each LUAD candidate. The current Top-10 is an expression-screening list, with unresolved compound identities, not an efficacy report.
2. Complete the full RECeSS nested-CV protocol before any SOTA-style claim. The three baseline algorithms were run with package defaults, not publication hyperparameters.
3. Run a frozen Jev decision evaluation if account access is available, then compare it with the saved rule/DeepSeek planner results. The Jev adapter exists, but no live Jev result is claimed.
4. Rehearse the updated nine-slide deck against the approximately 10-minute limit and add a demo video only if the instructor requests one. The reviewed course PDF requires GitHub code and an approximately 10-minute presentation, but does not prescribe a separate report, demo video, cover, or page count.

The current benchmark results support a negative finding for pure transcriptomic reversal on TRANSCRIPT. They do not establish clinical efficacy of any candidate drug.
