# System card

## System and current capability

The current implementation runs deterministic expression ranking, reproduces a limited TRANSCRIPT external benchmark, and packages a LUAD screening case with source hashes, quality checks, an identity audit, per-candidate evidence ledgers, a trace, and an external-model cost record. It is a research prototype. An optional [Jev adapter](jev_integration.md) exists but has no live results or account access; no general LLM planner is active. The benchmark strict mode reads item and user expression features for ranking; the benchmark adapter sees labels only through its training fold.

## Inputs and outputs

Inputs are versioned expression matrices and manifests listed in the [data card](data_card.md). The strict CLI writes score matrices, QC, trace, and manifest. The LUAD screen writes all 4,920 candidate scores, positive-control ranks, Top-10 ledgers, and an identity audit. `scripts/build_luad_case.py` verifies source and output checksums, consecutive ranks, ledger consistency, and the preregistered control list before writing `artifacts/reports/luad_case/case_report.json`. It routes the next research task to manual review by default; an explicit `--use-jev` enables advisory routing through the optional adapter. The case status remains `expression_screen_complete_evidence_pending`.

## Evaluation

See [the protocol](evaluation_protocol.md) and [benchmark results](benchmark_results.md). Three-seed random and weakly correlated splits use the official `stanscofi` global AUC/NDCG functions. ALSWR, PMF, and LogisticMF are direct package-default reruns; there is no publication-style nested hyperparameter search. Unknown associations count as non-positive in the official global metric convention, but are not known clinical failures. The LUAD frozen-control check reports ranks only for present names and does not turn absent names into negatives.

## Safeguards

The LUAD case validation rejects changed hashes, non-finite scores, broken ranks, mismatched ledgers, and changes to the control list. All unreviewed Top-10 candidates remain in the `insufficient_evidence` tier. The case report explicitly separates a transcriptomic score from clinical evidence and records unresolved drug identities. These checks cannot verify biological validity, citation relevance, treatment efficacy, or safety.

## Known gaps

Required before a clinical evidence report or full project completion: compound-level reconciliation, target/pathway and full literature review with support and conflict evidence (two context abstracts have been triaged), larger internal Eval suite, complete benchmark tuning and ablations, Jev access or documented fallback comparison, and the course-specific report, slides, and demo after the assignment PDF is supplied. No clinical deployment is supported.
