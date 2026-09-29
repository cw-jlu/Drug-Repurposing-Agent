# Drug Repurposing Agent

An auditable agent for transcriptomic drug repurposing, biomedical evidence retrieval, and structured decision routing with Jev.

The first comparison baseline is frozen as `v0.1.0`; its scope, metrics, API boundaries, and comparison rules are recorded in [the release note](docs/releases/v0.1.0.md).

The repository contains a controlled natural-language task router, a deterministic expression-ranking core, a strict-mode CLI, a RECeSS-compatible five-method adapter, three nested-tuned official benchmark baselines, a bounded LUAD A549 expression screen, and internal contract tests. The current computational study and course deliverables are complete; wet-lab validation and live Jev evaluation remain outside the completed scope.

A pilot check on TRANSCRIPT (`pilot_transcript/`, our own simplified evaluation, not the official RECeSS protocol) found that pure signature reversal scores AUC ≈ 0.48, no better than random, while a training-fold drug popularity baseline scores ≈ 0.73. Section 23 of the plan covers these results and the revised benchmark strategy.

Read the [full project plan](./药物重定位Agent项目计划.md) for the data sources, file formats, preprocessing steps, benchmark protocol, agent architecture, evaluation metrics, milestones, and known limitations.

The external evaluation uses the public [RECeSS drug repurposing benchmark](https://github.com/RECeSS-EU-Project/benchmark-code) and its [TRANSCRIPT v2.0.0 dataset](https://zenodo.org/records/7982976). A separate lung adenocarcinoma screen uses [GSE32863](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32863) and a [GSE92742-derived ExperimentHub subset](https://bioconductor.org/packages/release/data/experiment/html/signatureSearchData.html).

Raw datasets are not included in this repository. The plan records their official sources and verification requirements.

## Run the expression workflow

Use Python 3.10 or newer. Install with `python -m pip install -e .` and run:

```powershell
python -m drug_repurposing_agent --items data/raw/TRANSCRIPT_dataset_v2.0.0/items.csv --users data/raw/TRANSCRIPT_dataset_v2.0.0/users.csv --output artifacts/transcript_run
```

Download the official TRANSCRIPT v2.0.0 archive from [Zenodo](https://zenodo.org/records/7982976), verify MD5 `67b5be71611361ca493303b052a4944c`, and extract it under `data/raw/`. The CLI reads no labels. It writes four CSV matrices, QC, input SHA-256 hashes, trace, and limitations. New runs also write a durable JSONL trace by default; see the [trace policy](docs/trace_policy.md). `rrf.csv` is a ranking signal, not a calibrated treatment probability.

Run `python -m pytest -q` for the internal contract suite. See [evaluation protocol](docs/evaluation_protocol.md), [initial benchmark results](docs/benchmark_results.md), and [limitations](docs/limitations.md) before interpreting results. The benchmark runner additionally needs `stanscofi` 2.0.1 and its plotting/UMAP dependencies in a Python 3.10 environment with NumPy 1.26; its legacy `cute-ranking` dependency is incompatible with NumPy 2.

## Run the controlled Agent

The same CLI accepts a natural-language request and routes it through an allow-listed tool schema. The default planner is a deterministic, offline fallback. A live DeepSeek function-calling planner and the provider-neutral `StructuredPlanner` adapter are also available. Benchmark labels, raw matrices, and file paths are never exposed to a planner.

```powershell
python -m drug_repurposing_agent `
  --question "请根据转录组筛选候选药物" `
  --items data/raw/TRANSCRIPT_dataset_v2.0.0/items.csv `
  --users data/raw/TRANSCRIPT_dataset_v2.0.0/users.csv `
  --output artifacts/agent_run
```

The Agent writes `agent_run.json` with the validated plan, every tool transition, missing-input or failure state, and output references. Use `--mode research_open` for the frozen LUAD case; strict mode blocks research-only tools.

For DeepSeek, provide the key only through the process environment and add `--planner deepseek`. Do not put a key in source code, a CLI argument, or a committed `.env` file.

```powershell
$env:DEEPSEEK_API_KEY = "<set-locally>"
python -m drug_repurposing_agent `
  --question "使用 TRANSCRIPT 表达矩阵生成药物重定位排名" `
  --planner deepseek `
  --mode benchmark_strict `
  --items data/raw/TRANSCRIPT_dataset_v2.0.0/items.csv `
  --users data/raw/TRANSCRIPT_dataset_v2.0.0/users.csv `
  --output artifacts/deepseek_transcript_run
```

The frozen 20-case routing comparison, 61-case regression suite, and untouched 100-case Agent v3 holdout are reported in [planner evaluation results](docs/planner_eval_results.md). On v3, the deterministic planner scored 90/100 and the unchanged live DeepSeek planner scored 98/100.

The B1k/B2 nested-CV runs are committed. Reproduce one lightweight nested run with:

```powershell
python benchmarks/recess_adapter/nested_cv.py `
  --data data/raw/TRANSCRIPT_dataset_v2.0.0 `
  --split random_simple `
  --seed 1234
```

ALSWR, PMF, and LogisticMF were separately tuned with three inner folds over five outer seeds for both official split families. Reproduce one run with:

```powershell
python -m benchmarks.recess_adapter.nested_cv_official `
  --data data/raw/TRANSCRIPT_dataset_v2.0.0 `
  --split random_simple `
  --seed 1234
```

B2 was also run in the pinned RECeSS publication runner for 100 seeds and five inner folds on each split. Against the authors' 11 published TRANSCRIPT models, its NS-AUC is 0.5222 on random simple (9/12) and 0.5019 on weakly correlated (8/12). The exact seed-matched table, raw B2 CSVs, method patch, and interpretation limits are in [the official comparison](docs/recess_official_comparison.md).

B3 (`benchmarks/recess_adapter/official_b3.py`) fixes an orientation mismatch: the official NS-AUC ranks diseases within each drug row and counts ties as losses, while B2 ranked within disease columns. B3 rank-fuses six training-free components within drug rows. Its configuration was frozen before official scoring (`configs/b3_row_fusion_v1.json`). In the same 100-seed official runner it scores 0.7234 on random simple (2/13, behind BNNR 0.7331) and 0.6919 on weakly correlated (3/13, behind MBiRW 0.7384 and HAN 0.7029); see [the B3 section](docs/recess_official_comparison.md#b3-row-oriented-fusion-2026-09-28).

A second, separately labelled round, B4, row-rank-averages B3 with a NumPy BNNR port. The port reproduces the published per-seed BNNR scores exactly on three official seeds. Frozen at `e5ff7e7` before scoring, B4 reaches 0.7453 on random simple (rank 1/13; 87/100 paired wins over BNNR) and 0.6585 on weakly correlated (rank 3/13). B4 ranks first on random simple but not on weakly correlated, where MBiRW (0.7384) leads. It is therefore not state of the art across both protocols.

An interactive Streamlit demo (`streamlit run app/demo.py`, see [docs/demo.md](docs/demo.md)) replays or runs the Agent and shows LUAD evidence cards and benchmark tables. Defense figures are regenerated by `python scripts/make_figures.py` into [docs/figures](docs/figures/README.md).

An [official-runner component ablation](docs/component_and_llm_adjudication.md) separates B2 into B0p, B1k and B1. B2 leads those four methods on random simple, but B1 alone leads on weakly correlated. One prescore DeepSeek choice selected these two methods before the ablation finished; a separate constrained DeepSeek call triaged LUAD research actions only. Neither single-call result establishes general LLM model-selection accuracy or drug efficacy.

On Windows with Python 3.10, `scripts/setup_benchmark.ps1` installs the tested environment from [the pinned dependency file](requirements-benchmark.lock) and runs the contract suite.

The LUAD disease-signature tool processed GSE32863/GPL6884 and identified [57 verifiable tumor/normal pairs](docs/luad_data_audit.md), with two unmatched samples excluded. A verified EH3226 A549 subset produced a [Top-10 transcriptomic screening report](docs/luad_screening_report.md). Deterministic reconstruction recovered one source GSE92742 signature and perturbagen ID for each Top-10 column; cross-source chemical discrepancies and the absence of efficacy evidence remain unresolved.

For the LUAD audit, `python scripts/fetch_luad_inputs.py` retrieves the GEO sources; `python scripts/process_gse32863.py` builds the paired disease signature; `python scripts/run_limma_sensitivity.py` runs the optional paired R/limma verification; `python scripts/filter_lincs_a549.py` freezes the A549 metadata; `python scripts/fetch_eh3226.py` retrieves the 2.46 GB expression subset; `python scripts/rank_luad_eh3226.py` ranks the candidates; `python scripts/fetch_broad_annotations.py` and `python scripts/audit_luad_identity.py` check Broad sample annotations; `python scripts/search_luad_pubmed.py` retrieves literature leads; `python scripts/curate_luad_literature.py` adds two carefully scoped context records; and `python scripts/build_luad_case.py` validates and packages the case report and trace. Install the `lincs` extra for the EH3226 step. The full 21.3 GB GEO Level 5 archive is unnecessary for this bounded screen.

The editable [course report v5](deliverables/药物重定位Agent_课程设计报告_v5.docx) and [eleven-slide defense deck v7](deliverables/药物重定位Agent_答辩稿_v7.pptx) (built by `scripts/build_defense_deck_v7.mjs` from committed results; outline in [docs/defense_rehearsal_v7.md](docs/defense_rehearsal_v7.md)) add the B3/B4 results, contamination probe, decision-layer ablation and multi-agent review. Earlier versions are in `deliverables/archive/` and include the official 100-run NS-AUC comparison, component ablation, trace-grade boundary, five-partition amended method-selection negative result, and wet-lab limit. Earlier versions remain as snapshots. A [ten-minute presentation guide](docs/defense_rehearsal_10min.md) is available. The five disease-partition [v2 method-selection protocol and outcome](docs/method_selection_partition_v2.md) documents the original splitter failure and separately amended test: DeepSeek selection scored 0.40653 against fixed B2 0.51684 mean NS-AUC. A [B1k metric audit](docs/b1k_metric_audit.md) measures strict-score ties across the frozen official seeds without changing the benchmark. The [v3 prospective protocol](docs/method_selection_v3_protocol.md) requires training-only evidence and a new independent holdout; its [external-input audit](docs/external_holdout_feasibility.md) has staged disease signatures but no label-ready evaluation, so v3 has not run. A prescore [Agent v4 evidence-card holdout](docs/agent_v4_claim_results.md) scored 14/20 strict cases with redacted provider-visible traces; it is not clinical efficacy validation. The existing two-choice selector has a [seed-paired retrospective audit](docs/benchmark_results.md). See [project status](docs/project_status.md), [delivery audit](docs/completion_audit.md), [data card](docs/data_card.md), [system card](docs/system_card.md), and the [experimental validation protocol](docs/luad_experimental_validation_protocol.md) for completed work and remaining limits.
