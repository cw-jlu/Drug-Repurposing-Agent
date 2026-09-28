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

| Frozen method | Random simple NS-AUC | Weakly correlated NS-AUC |
| --- | ---: | ---: |
| B0p: training association count | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 |
| B1k: expression-neighbor propagation | 0.1747 ± 0.0393 | 0.0562 ± 0.0466 |
| B1: label-free expression reversal | 0.4816 ± 0.0340 | **0.5417 ± 0.0000** |
| B2: fixed RRF fusion | **0.5222 ± 0.0354** | 0.5019 ± 0.0036 |

Values are means ± sample SD over 100 upstream metric values. On random
simple, B2 exceeds B1 by 0.0407 paired NS-AUC points and wins 97/100 seeds;
on weakly correlated, B1 exceeds B2 by 0.0398 and wins 100/100 seeds. The
weak-split B1 zero SD reflects the same deterministic outer holdout, **not**
evidence of negligible uncertainty. The B1k adapter was additionally checked
against the frozen `TranscriptBaseline` on the real TRANSCRIPT first random
split; all three component score vectors matched exactly (maximum absolute
difference 0). The very low B1k NS-AUC is therefore not explained by an
adapter score mismatch, though the dataset/metric cause has not been proven.
The versioned aggregate CSVs and parameter files are in
`benchmark/results/recess_official_components/`.

The B0p random run was resumed after changing only `njobs` from 1 to 4:
the official runner reused the first 14 completed seed intermediates and
computed the rest in parallel. Both parameter receipts are retained. Seed
order, model code, folds and metric were unchanged; `njobs` affects execution
scheduling only.

Before inspecting component ablation results, one accepted DeepSeek Flash call chose a
method from the four frozen options using only the versioned method, split,
and dataset descriptions in `configs/component_selector_v1.json`. It selected
`B2` for random simple and `B1` for weakly correlated. The choices, input
hash, timestamp and token usage were committed in
`benchmark/results/recess_component_llm_selector_v1.json` at `fbd9858` while
the component runs were still in progress. An earlier strict-tool response
failed the local reason-length check and was not saved; the accepted retry
still preceded inspection of ablation scores. This is a single prescore
choice, not a tuned classifier or a reliable estimate of method-selection
skill. After the run, both choices rank first among these **four** methods:
selected NS-AUC
is 0.5222 for random simple and 0.5417 for weakly correlated. Relative to
always using B2, the latter is +0.0398 on the same repeated outer holdout.
There are only two split-level decisions, no candidate- or disease-specific
LLM routing and no independent model-selection holdout. This result cannot
support a claim that LLM selection generally improves prediction or beats
the published 11-model panel.
The model's random-split rationale described B0p/B1k label signal as reliable,
but their observed NS-AUCs were 0.5000 and 0.1747. Thus even its successful
method choice came with a partly unsupported explanation. The reasoning text
must not be treated as an interpretation of the measured result.

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
