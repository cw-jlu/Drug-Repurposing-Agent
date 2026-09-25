# System card

## System and current capability

The current implementation accepts a natural-language research request, produces a validated allow-listed tool plan, runs deterministic expression ranking, reproduces a limited TRANSCRIPT external benchmark, and packages a LUAD screening case with source hashes, quality checks, an identity audit, per-candidate evidence ledgers, a trace, and an external-model cost record. It is a research prototype. The default planner is a deterministic fallback; a live DeepSeek function-calling planner and a provider-neutral structured-LLM adapter are available. An optional [Jev adapter](jev_integration.md) exists without live results or account access. The benchmark strict mode reads item and user expression features for ranking; the planner receives only the question, mode, safe input inventory, and tool schemas, while the benchmark adapter sees labels only through its training fold.

## Inputs and outputs

Inputs are versioned expression matrices and manifests listed in the [data card](data_card.md). The strict CLI writes score matrices, QC, trace, and manifest. The LUAD screen writes all 4,920 candidate scores, positive-control ranks, Top-10 ledgers, and an identity audit. `scripts/build_luad_case.py` verifies source and output checksums, consecutive ranks, ledger consistency, and the preregistered control list before writing `artifacts/reports/luad_case/case_report.json`. It routes the next research task to manual review by default; an explicit `--use-jev` enables advisory routing through the optional adapter. The case status remains `expression_screen_complete_evidence_pending`.

## Evaluation

See [the protocol](evaluation_protocol.md) and [benchmark results](benchmark_results.md). B1k/B2 use three outer seeds and three-fold inner tuning. ALSWR, PMF, and LogisticMF use five outer seeds, two split protocols, a prespecified four-candidate grid, and three-fold inner tuning. A separate B2 run in the publication runner uses the same 100 seeds and five-fold model selection as 11 published TRANSCRIPT models; its NS-AUC ranks are 9/12 and 8/12 across the two splits. Every project nested result includes an explicit zero-overlap audit. Unknown associations count as non-positive in the global metric convention, but are not known clinical failures. The independent 100-case Agent v3 holdout was frozen before provider calls; the rule and live DeepSeek planners scored 90% and 98%, respectively.

## Safeguards

The LUAD case validation rejects changed hashes, non-finite scores, broken ranks, mismatched ledgers, and changes to the control list. All Top-10 candidates remain in the `insufficient_evidence` tier. Exact source GSE92742 IDs are recorded separately from the cross-source Broad chemical-identity audit. These checks cannot verify biological validity, citation relevance, treatment efficacy, or safety.

## Known gaps

Required before any clinical efficacy claim: execute the preregistered compound procurement, dose, subtype, mechanism, and combination experiments; independently replicate the findings; and assess safety. Further benchmark search spaces, live Jev evaluation (if access becomes available), and broader Agent/provider repetitions remain optional research extensions. The Top-10 source-ID reconstruction, evidence matrix, independent v3 holdout, updated report, and updated course slides are complete. No clinical deployment is supported.
