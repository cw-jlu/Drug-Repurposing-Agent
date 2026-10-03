"""Real tool backend for agent v2, wired to the project's deterministic code.

The disease is resolved from the request by ``geo_cohort.resolve_disease`` against
configs/disease_registry_v1.json; only registered diseases can be analysed.
``fetch_geo_series`` downloads the pinned GEO files and verifies SHA-256; the
cohort and signature tools always recompute from the raw series matrix (no
processed/cached outputs are read). Heavy modules are imported lazily from the
repository's scripts/ and evals/ packages, so the command must run from the
repository root. Every tool returns {"summary": {...}, ...artefact data} and
raises on failure; the executor records it. The literature tool reuses the frozen
multi-agent review when the Top-10 matches it and refuses otherwise, so a demo run
never triggers unplanned paid model calls. Other registered diseases have no frozen
review: their literature review runs live only, with the registry's review config.
"""

from __future__ import annotations

import json
from pathlib import Path

from .agent_v2 import Step, ToolError
from .geo_cohort import CohortError, cohort, fetch, files_ready, load_registry, resolve_disease, signature
from .trace import TraceRecorder
from .workflow import Mode

ROOT = Path(".")
TRANSCRIPT = ROOT / "data/raw/TRANSCRIPT_dataset_v2.0.0"
DRUG_H5 = ROOT / "data/raw/GSE92742/lincs_EH3226.h5"
REVIEW = ROOT / "benchmark/results/multi_agent_review_v1.json"
LUAD_ID = "luad_gse32863"


def available_inputs(question: str | None = None) -> tuple[str, ...]:
    """Inputs the executor holds for this request.

    With a question, disease inputs are offered only for the registered disease the
    question names: ``registered_disease`` (it can be fetched) and ``disease_series``
    (its raw files are already on disk). Without a question (machine inventory), any
    registered disease whose files are present yields ``disease_series``.
    """
    registry = load_registry()
    have = []
    if question is None:
        entries = list(registry.values())
    else:
        entry = resolve_disease(question, registry)
        entries = [entry] if entry else []
        if entry:
            have.append("registered_disease")
    if any(files_ready(e) for e in entries):
        have.append("disease_series")
    if DRUG_H5.is_file():
        have.append("drug_signatures")
    if (TRANSCRIPT / "items.csv").is_file() and (TRANSCRIPT / "users.csv").is_file():
        have += ["items", "users"]
    return tuple(have)


def real_backend(output: Path, mode: Mode, disease: dict | None = None, live_review: bool = False,
                 max_review_calls: int = 60) -> callable:
    trace = TraceRecorder("agent_v2_tools", output / "traces")
    trace.emit("disease_resolved", disease=disease["id"] if disease else None,
               accession=disease["accession"] if disease else None)

    def need_disease() -> dict:
        if disease is None:
            raise ToolError("the request names no registered disease (see configs/disease_registry_v1.json)")
        return disease

    def fetch_series(_: dict) -> dict:
        try:
            result = fetch(need_disease(), trace)
        except (CohortError, OSError) as exc:
            raise ToolError(f"GEO download failed: {exc}") from exc
        return {"summary": result}

    def qc(_: dict) -> dict:
        entry = need_disease()
        try:
            samples = cohort(entry)
        except CohortError as exc:
            raise ToolError(str(exc)) from exc
        included = int(samples.included.sum())
        kept = samples.loc[samples.included, "condition"]
        groups = ({"pairs": included // 2} if entry.get("design", "paired") == "paired" else
                  {"tumor": int((kept == "Tumor").sum()), "normal": int((kept == "Normal").sum())})
        return {"samples": samples, "summary": {"disease": entry["label"], "accession": entry["accession"],
                                                "design": entry.get("design", "paired"), "samples": len(samples),
                                                "included": included, **groups,
                                                "excluded": len(samples) - included}}

    def deg(a: dict) -> dict:
        try:
            table, summary = signature(need_disease(), a["cohort"]["samples"])
        except CohortError as exc:
            raise ToolError(str(exc)) from exc
        trace.emit("signature_recomputed", **summary)
        return {"deg": table.dropna(subset=["log2FC", "fdr"]), "summary": summary}

    def pathways(a: dict) -> dict:
        from scripts.luad_pathway_enrichment import load_library, ora
        table = a["signature"]["deg"]
        up = set(table.loc[(table.fdr < 0.05) & (table.log2FC >= 1), "gene_symbol"])
        down = set(table.loc[(table.fdr < 0.05) & (table.log2FC <= -1), "gene_symbol"])
        lib = load_library("MSigDB_Hallmark_2020", trace)
        background = set(table.gene_symbol) & set().union(*map(set, lib.values()))
        top = lambda rows: [r["term"] for r in rows if r.get("fdr", 1) < 0.05][:5]
        return {"summary": {"up": top(ora(up, background, lib)), "down": top(ora(down, background, lib))}}

    def rank(a: dict) -> dict:
        import pandas as pd
        from drug_repurposing_agent.data import ExpressionData
        from drug_repurposing_agent.ranking import score_expressions
        from evals.luad_threshold_sensitivity import rank as rank_fn
        from scripts.luad_pathway_reversal import _load_drugs
        cell = need_disease()["drug_cell_line"]
        table = a["signature"]["deg"].set_index("gene_symbol")
        drugs = _load_drugs(table, trace, cell_line=cell)
        if drugs.shape[1] == 0:
            raise ToolError(f"no LINCS compound signatures for cell line {cell}")
        genes = drugs.index.tolist()
        disease_vec = pd.DataFrame({"disease": table.loc[genes, "log2FC"].to_numpy()}, index=genes)
        reversal = score_expressions(ExpressionData(drugs, disease_vec))["spearman_reversal"]["disease"].to_numpy()
        lm = table.loc[genes]
        up = set(lm.index[(lm.fdr < 0.05) & (lm.log2FC >= 1)])
        down = set(lm.index[(lm.fdr < 0.05) & (lm.log2FC <= -1)])
        ranks = rank_fn(drugs, reversal, up, down)
        return {"ranks": ranks, "summary": {"cell_line": cell, "drugs": len(ranks), "top10": ranks.index[:10].tolist()}}

    def audit(a: dict) -> dict:
        import pandas as pd
        from evals.luad_positive_control_stats import statistics
        entry = need_disease()
        ranks = a["ranking"]["ranks"]
        summary = {"top10_matches_frozen_identity_audit": None}
        if entry.get("evidence_records"):
            frozen = [c["name"] for c in json.loads(Path(entry["evidence_records"]).read_text(encoding="utf-8"))["candidates"]]
            summary["top10_matches_frozen_identity_audit"] = ranks.index[:10].tolist() == frozen
        if not entry.get("reference_drugs"):
            summary["reference_drugs"] = "none prespecified for this disease; recovery statistic not computed"
            return {"summary": summary}
        controls = pd.read_csv(entry["reference_drugs"]).drug_name.tolist()
        measured = {c: int(ranks[c]) for c in controls if c in ranks.index}
        stats = statistics(measured, n_names=len(ranks), draws=20000)
        summary.update({"measured_controls": len(measured), "mean_percentile": round(stats["mean_percentile"], 3),
                        "permutation_p": round(stats["permutation_p_one_sided"], 3)})
        return {"summary": summary}

    def review(a: dict) -> dict:
        entry = need_disease()
        if not entry.get("literature"):
            raise ToolError("no literature-review configuration for this disease in the registry")
        top10 = a["ranking"]["ranks"].index[:10].tolist()
        if live_review:
            return live(top10, entry)
        if entry["id"] != LUAD_ID:
            raise ToolError(f"no frozen literature review exists for {entry['label']}; "
                            "it can only run live (PubMed + DeepSeek), which is not enabled")
        frozen = json.loads(REVIEW.read_text(encoding="utf-8"))
        if [c["name"] for c in frozen["candidates"]] != top10:
            raise ToolError("Top-10 differs from the frozen review; a live multi-agent review is not enabled")
        tiers = {c["name"]: c["tier"] for c in frozen["candidates"]}
        return {"tiers": tiers, "summary": {"source": "frozen multi_agent_review_v1 (reused)",
                                            "tier_counts": frozen["tier_counts"],
                                            "rejected_citations": frozen["citation_validation"]["rejected_quoted_items"]}}

    def live(top10: list[str], entry: dict) -> dict:
        """Run the multi-agent review now (PubMed + DeepSeek) with the disease's review config.

        LUAD candidates must have identity records (the frozen evidence file); for other
        diseases each candidate is reviewed by its LINCS name and marked as not identity-audited.
        """
        from .deepseek import local_api_key
        from .llm_calls import CallStats, TracedToolCaller
        from .multi_agent_review import AbstractPubMedClient, MultiAgentReviewer, review_config
        config = review_config(entry["literature"])
        if entry.get("evidence_records"):
            records = {c["name"]: c for c in
                       json.loads(Path(entry["evidence_records"]).read_text(encoding="utf-8"))["candidates"]}
            missing = [n for n in top10 if n not in records]
            if missing:
                raise ToolError(f"no evidence identity record for {missing}; live review refused")
        else:
            records = {n: {"rank": i, "name": n, "pathways": [],
                           "identity": "not identity-audited: LINCS compound name only"}
                       for i, n in enumerate(top10, 1)}
        key = local_api_key()
        trace.add_secret(key)
        stats = CallStats()
        reviewer = MultiAgentReviewer(TracedToolCaller(key, stats=stats), AbstractPubMedClient(), trace,
                                      config=config)
        results = []
        for name in top10:
            if stats.as_dict()["attempts"] + 3 > max_review_calls:
                raise ToolError("live review call budget would be exceeded")
            results.append(reviewer.review(records[name]))
        tiers = {r["name"]: r["tier"] for r in results}
        frozen_tiers = ({c["name"]: c["tier"] for c in json.loads(REVIEW.read_text(encoding="utf-8"))["candidates"]}
                        if entry["id"] == LUAD_ID else {})
        rejected = sum(r["rejected_claims"]["support"] + r["rejected_claims"]["contradiction"] for r in results)
        proposed = sum(r["proposed_claims"]["support"] + r["proposed_claims"]["contradiction"] for r in results)
        calls = stats.as_dict()
        return {"tiers": tiers, "results": results,
                "summary": {"source": "live multi-agent review (PubMed + DeepSeek)", "disease": config.disease,
                            "calls": calls["attempts"],
                            "tier_counts": {t: list(tiers.values()).count(t) for t in sorted(set(tiers.values()))},
                            "agrees_with_frozen": (sum(tiers[n] == frozen_tiers.get(n) for n in tiers)
                                                   if frozen_tiers else "no frozen review for this disease"),
                            "proposed_quoted_items": proposed, "rejected_quoted_items": rejected}}

    def report(a: dict) -> dict:
        entry = need_disease()
        body = {"disease": entry["label"], "accession": entry["accession"], "drug_cell_line": entry["drug_cell_line"],
                "top10": a["ranking"]["summary"]["top10"],
                "signature": a.get("signature", {}).get("summary", "not computed"),
                "pathways": a.get("pathways", {}).get("summary", "not computed"),
                "audit": a.get("audit", {}).get("summary", "not computed"),
                "evidence": a.get("evidence", {}).get("tiers", "literature review not run"),
                "boundary": "transcriptomic research hypotheses only; not treatment advice"}
        path = output / "candidate_report.json"
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        return {"summary": {"report": str(path), "evidence_included": "evidence" in a}}

    def transcriptome(_: dict) -> dict:
        from .workflow import run_expression_workflow
        out = output / "expression_ranking"
        result = run_expression_workflow(TRANSCRIPT / "items.csv", TRANSCRIPT / "users.csv", out, mode, 100)
        return {"summary": {"status": result["status"], "manifest": str(out / "manifest.json")}}

    table = {"fetch_geo_series": fetch_series, "qc_disease_cohort": qc, "differential_expression": deg,
             "pathway_enrichment": pathways, "rank_candidates": rank, "audit_candidates": audit,
             "review_literature": review, "build_report": report, "rank_transcriptome": transcriptome}

    def run(step: Step, artefacts: dict) -> dict:
        trace.emit("tool_call", tool=step.tool)
        return table[step.tool](artefacts)
    return run


def default_luad() -> dict:
    """The LUAD registry entry, for callers whose frozen cases are LUAD by construction."""
    return load_registry()[LUAD_ID]
