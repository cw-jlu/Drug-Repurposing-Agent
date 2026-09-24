# Drug Repurposing Agent

An auditable agent for transcriptomic drug repurposing, biomedical evidence retrieval, and structured decision routing with Jev.

The repository contains a deterministic expression-ranking core, a strict-mode CLI, a RECeSS-compatible five-method adapter, three direct benchmark baselines, a bounded LUAD A549 expression screen, and internal contract tests. The planned clinical evidence study and course deliverables are not yet complete.

A pilot check on TRANSCRIPT (`pilot_transcript/`, our own simplified evaluation, not the official RECeSS protocol) found that pure signature reversal scores AUC ≈ 0.48, no better than random, while a training-fold drug popularity baseline scores ≈ 0.73. Section 23 of the plan covers these results and the revised benchmark strategy.

Read the [full project plan](./药物重定位Agent项目计划.md) for the data sources, file formats, preprocessing steps, benchmark protocol, agent architecture, evaluation metrics, milestones, and known limitations.

The external evaluation uses the public [RECeSS drug repurposing benchmark](https://github.com/RECeSS-EU-Project/benchmark-code) and its [TRANSCRIPT v2.0.0 dataset](https://zenodo.org/records/7982976). A separate lung adenocarcinoma screen uses [GSE32863](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32863) and a [GSE92742-derived ExperimentHub subset](https://bioconductor.org/packages/release/data/experiment/html/signatureSearchData.html).

Raw datasets are not included in this repository. The plan records their official sources and verification requirements.

## Run the expression workflow

Use Python 3.10 or newer. Install with `python -m pip install -e .` and run:

```powershell
python -m drug_repurposing_agent --items data/raw/TRANSCRIPT_dataset_v2.0.0/items.csv --users data/raw/TRANSCRIPT_dataset_v2.0.0/users.csv --output artifacts/transcript_run
```

Download the official TRANSCRIPT v2.0.0 archive from [Zenodo](https://zenodo.org/records/7982976), verify MD5 `67b5be71611361ca493303b052a4944c`, and extract it under `data/raw/`. The CLI reads no labels. It writes four CSV matrices, QC, input SHA-256 hashes, trace, and limitations. `rrf.csv` is a ranking signal, not a calibrated treatment probability.

Run `python -m pytest -q` for the internal contract suite. See [evaluation protocol](docs/evaluation_protocol.md), [initial benchmark results](docs/benchmark_results.md), and [limitations](docs/limitations.md) before interpreting results. The benchmark runner additionally needs `stanscofi` 2.0.1 and its plotting/UMAP dependencies in a Python 3.10 environment with NumPy 1.26; its legacy `cute-ranking` dependency is incompatible with NumPy 2.

On Windows with Python 3.10, `scripts/setup_benchmark.ps1` installs the tested environment from [the pinned dependency file](requirements-benchmark.lock) and runs the contract suite.

The LUAD disease-signature tool processed GSE32863/GPL6884 and identified [57 verifiable tumor/normal pairs](docs/luad_data_audit.md), with two unmatched samples excluded. A verified EH3226 A549 subset produced a [Top-10 transcriptomic screening report](docs/luad_screening_report.md). Its candidates have insufficient independent evidence and several unresolved drug identities.

For the LUAD audit, `python scripts/fetch_luad_inputs.py` retrieves the GEO sources; `python scripts/process_gse32863.py` builds the paired disease signature; `python scripts/run_limma_sensitivity.py` runs the optional paired R/limma verification; `python scripts/filter_lincs_a549.py` freezes the A549 metadata; `python scripts/fetch_eh3226.py` retrieves the 2.46 GB expression subset; `python scripts/rank_luad_eh3226.py` ranks the candidates; `python scripts/fetch_broad_annotations.py` and `python scripts/audit_luad_identity.py` check Broad sample annotations; `python scripts/search_luad_pubmed.py` retrieves literature leads; `python scripts/curate_luad_literature.py` adds two carefully scoped context records; and `python scripts/build_luad_case.py` validates and packages the case report and trace. Install the `lincs` extra for the EH3226 step. The full 21.3 GB GEO Level 5 archive is unnecessary for this bounded screen.

The editable [course report draft](deliverables/药物重定位Agent_课程设计报告草稿.docx) and [nine-slide defense draft](deliverables/药物重定位Agent_答辩草稿_v2.pptx) summarize the verified results. Both await the missing course-assignment PDF for final structure and formatting. See [project status](docs/project_status.md), [delivery audit](docs/completion_audit.md), [data card](docs/data_card.md), [system card](docs/system_card.md), and the [optional Jev adapter notes](docs/jev_integration.md) for completed work and remaining deliverables.
