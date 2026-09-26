# Decomposed B2 prediction and bounded LLM research triage

These are two distinct experiments. The first measures drug–disease ranking
accuracy on TRANSCRIPT. The second asks an LLM to prioritize **next research
actions** for the frozen LUAD Top-10. Its output is not a drug efficacy score and
must not be merged with the NS-AUC table.

## B2 component ablation

The frozen B2 model is an unweighted reciprocal-rank fusion of three inputs:
`B0p` (training-fold positive count per drug), `B1k` (drug-expression kNN
propagation of training-fold positive labels), and `B1` (negative Spearman
correlation between drug and disease expression signatures). The component
adapters in `benchmarks/recess_adapter/official_components.py` exactly match
the corresponding frozen `TranscriptBaseline` scores in unit tests; changing
validation ratings leaves their scores unchanged. `B0p` and `B1k` use only
training-fold associations.

The adapters enter the same pinned [RECeSS official runner](https://github.com/RECeSS-EU-Project/benchmark-code)
as B2, using 100 identical seed values, a 20% outer holdout, five inner folds,
and the author's NS-AUC calculation for both split types. The second patch is
applied *after* `official_b2.patch` and does not modify B2's implementation.
The published five-fold operation selects the best fold-trained model by inner
row-wise AUC; it does not search a hyperparameter grid. The weakly correlated
split repeats one outer holdout across seeds, so its spread is not uncertainty
over 100 distinct outer splits.

To reproduce, apply both patches to a clean clone at commit
`a7f11077271cedf3a98a82e3dc74b6fc0e93986e`, then run:

```powershell
python -m scripts.run_recess_official_components --upstream $upstream --n 100 --k 5 --njobs 4
python -m scripts.compare_recess_components
```

The comparison checks all 100 seed positions, finite metrics, pairwise B2
differences, and source hashes. It writes
`benchmark/results/recess_official_component_ablation.json`.

## LLM evidence triage

`scripts/run_luad_llm_adjudication.py` reads the frozen
`configs/luad_top10_evidence_v1.json` and the human-reviewed
`docs/luad_top10_evidence_matrix.md`. It does not read TRANSCRIPT labels or
benchmark predictions. DeepSeek Flash is asked for one to three **research
actions**, each tied to an existing rank and supplied PMID. A strict function
schema and a local validator reject unknown drugs, fabricated/mis-scoped PMIDs,
unknown actions, duplicate ranks, clinical actions, and an assay recommendation
for a cross-source identity-unresolved compound. All ten evidence tiers remain
`insufficient_evidence`; model prose is advisory and unverified.

The real call used the [DeepSeek strict tool-call mode](https://api-docs.deepseek.com/guides/tool_calls/)
on 2026-09-26. Two earlier JSON-mode responses failed local validation and
were not accepted or saved. The one accepted response prioritized rank 6
`BRD-K84203638` and rank 5 `flumetasone` for identity resolution, and rank 2
`beclomethasone-dipropionate` for literature review. It did not propose a
treatment or claim efficacy. The validated output and input hashes are saved in
`benchmark/results/luad_llm_adjudication_v1.json`. This is **one model sample**,
not a blinded accuracy estimate or proof that its prioritization is optimal.

Run it again only with a local `DEEPSEEK_API_KEY` or an ignored `.env` file:

```powershell
python -m scripts.run_luad_llm_adjudication
```

The key is never written to the result. Repeated model calls and blinded human
review would be needed to quantify decision consistency and evidence quality.

## Planner v3 is a separate metric

Agent v3's 98/100 is exact-match selection of the expected tool on a frozen
100-case set; it does not grade drug predictions. Its two failures are
`v3_luad_003` (`package_luad_case` expected, `manual_review` selected) and
`v3_boundary_004` (`manual_review` expected, `rank_transcriptome` selected).
Both were valid tool calls, not API errors. The stored trace has no raw model
reasoning, so their internal cause cannot be determined from this run. See
`docs/planner_eval_results.md`.
