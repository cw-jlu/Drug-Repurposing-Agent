# System card

## System and current capability

The current implementation accepts a natural-language research request, produces a validated allow-listed tool plan, runs deterministic expression ranking, reproduces a limited TRANSCRIPT external benchmark, and packages a LUAD screening case with source hashes, quality checks, an identity audit, per-candidate evidence ledgers, a trace, and an external-model cost record. It is a research prototype. The default planner is a deterministic fallback; a live DeepSeek function-calling planner and a provider-neutral structured-LLM adapter are available. An optional [Jev adapter](jev_integration.md) exists without live results or account access. The benchmark strict mode reads item and user expression features for ranking; the planner receives only the question, mode, safe input inventory, and tool schemas, while the benchmark adapter sees labels only through its training fold.

## Inputs and outputs

Inputs are versioned expression matrices and manifests listed in the [data card](data_card.md). The strict CLI writes score matrices, QC, trace, and manifest. The LUAD screen writes all 4,920 candidate scores, positive-control ranks, Top-10 ledgers, and an identity audit. `scripts/build_luad_case.py` verifies source and output checksums, consecutive ranks, ledger consistency, and the preregistered control list before writing `artifacts/reports/luad_case/case_report.json`. It routes the next research task to manual review by default; an explicit `--use-jev` enables advisory routing through the optional adapter. The case status remains `expression_screen_complete_evidence_pending`.

## Evaluation

See [the protocol](evaluation_protocol.md) and [benchmark results](benchmark_results.md). Three-seed random and weakly correlated splits use the RECeSS/stanscofi split and global-metric definitions. B1k/B2 additionally use three-fold inner tuning with an explicit zero-overlap audit. ALSWR, PMF, and LogisticMF remain direct package-default reruns rather than publication-style tuned models. Unknown associations count as non-positive in the official global metric convention, but are not known clinical failures. The LUAD frozen-control check reports ranks only for present names and does not turn absent names into negatives.

## Safeguards

The LUAD case validation rejects changed hashes, non-finite scores, broken ranks, mismatched ledgers, and changes to the control list. All unreviewed Top-10 candidates remain in the `insufficient_evidence` tier. The case report explicitly separates a transcriptomic score from clinical evidence and records unresolved drug identities. These checks cannot verify biological validity, citation relevance, treatment efficacy, or safety.

## Known gaps

Required before a clinical efficacy report or full project completion: compound-level reconciliation and experimental validation, broader benchmark tuning/ablations, a new untouched Agent holdout after v2 hardening, live Jev evaluation if access becomes available, and updated course slides for the DeepSeek/nested-CV comparison. The bounded Top-10 target/pathway and support/conflict literature matrix is complete, but it cannot establish efficacy. No clinical deployment is supported.
