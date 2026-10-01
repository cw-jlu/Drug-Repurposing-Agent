"""Real tool backend for agent v2, wired to the project's deterministic code.

Heavy modules are imported lazily from the repository's scripts/ and evals/ packages,
so the command must run from the repository root. Every tool returns
{"summary": {...}, ...artefact data} and raises on failure; the executor records it.
The literature tool reuses the frozen multi-agent review when the Top-10 matches it
and refuses otherwise, so a demo run never triggers unplanned paid model calls.
"""

from __future__ import annotations

import json
from pathlib import Path

from .agent_v2 import Step, ToolError
from .trace import TraceRecorder
from .workflow import Mode

ROOT = Path(".")
TRANSCRIPT = ROOT / "data/raw/TRANSCRIPT_dataset_v2.0.0"
REVIEW = ROOT / "benchmark/results/multi_agent_review_v1.json"
EVIDENCE = ROOT / "configs/luad_top10_evidence_v1.json"


def available_inputs() -> tuple[str, ...]:
    have = []
    if (ROOT / "data/raw/GSE32863/GSE32863_series_matrix.txt.gz").is_file():
        have.append("disease_series")
    if (ROOT / "data/raw/GSE92742/lincs_EH3226.h5").is_file():
        have.append("drug_signatures")
    if (TRANSCRIPT / "items.csv").is_file() and (TRANSCRIPT / "users.csv").is_file():
        have += ["items", "users"]
    return tuple(have)


def real_backend(output: Path, mode: Mode) -> callable:
    trace = TraceRecorder("agent_v2_tools", output / "traces")

    def qc(_: dict) -> dict:
        from scripts import process_gse32863 as proc
        _, metadata = proc.table_start(proc.SERIES, "!series_matrix_table_begin")
        samples = proc.sample_manifest(metadata)
        included = int(samples.included.sum())
        if included < 20:
            raise ToolError(f"only {included} paired samples")
        return {"samples": samples, "summary": {"samples": len(samples), "included": included,
                                                "pairs": included // 2, "excluded": len(samples) - included}}

    def deg(_: dict) -> dict:
        from scripts.make_figures import load_deg
        table, source = load_deg(trace)
        table = table.dropna(subset=["log2FC", "fdr"])
        up = int(((table.fdr < 0.05) & (table.log2FC >= 1)).sum())
        down = int(((table.fdr < 0.05) & (table.log2FC <= -1)).sum())
        return {"deg": table, "summary": {"genes": len(table), "up": up, "down": down, "source": source}}

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
        from drug_repurposing_agent.data import ExpressionData
        from drug_repurposing_agent.ranking import score_expressions
        from evals.luad_threshold_sensitivity import rank as rank_fn
        from scripts.luad_pathway_reversal import _load_drugs
        table = a["signature"]["deg"].set_index("gene_symbol")
        drugs = _load_drugs(table, trace)
        genes = drugs.index.tolist()
        import pandas as pd
        disease = pd.DataFrame({"LUAD": table.loc[genes, "log2FC"].to_numpy()}, index=genes)
        reversal = score_expressions(ExpressionData(drugs, disease))["spearman_reversal"]["LUAD"].to_numpy()
        lm = table.loc[genes]
        up = set(lm.index[(lm.fdr < 0.05) & (lm.log2FC >= 1)])
        down = set(lm.index[(lm.fdr < 0.05) & (lm.log2FC <= -1)])
        ranks = rank_fn(drugs, reversal, up, down)
        return {"ranks": ranks, "summary": {"drugs": len(ranks), "top10": ranks.index[:10].tolist()}}

    def audit(a: dict) -> dict:
        import pandas as pd
        from evals.luad_positive_control_stats import statistics
        ranks = a["ranking"]["ranks"]
        controls = pd.read_csv(ROOT / "configs/luad_positive_controls.csv").drug_name.tolist()
        measured = {c: int(ranks[c]) for c in controls if c in ranks.index}
        stats = statistics(measured, n_names=len(ranks), draws=20000)
        frozen = [c["name"] for c in json.loads(EVIDENCE.read_text(encoding="utf-8"))["candidates"]]
        return {"summary": {"measured_controls": len(measured), "mean_percentile": round(stats["mean_percentile"], 3),
                            "permutation_p": round(stats["permutation_p_one_sided"], 3),
                            "top10_matches_frozen_identity_audit": ranks.index[:10].tolist() == frozen}}

    def review(a: dict) -> dict:
        top10 = a["ranking"]["ranks"].index[:10].tolist()
        frozen = json.loads(REVIEW.read_text(encoding="utf-8"))
        if [c["name"] for c in frozen["candidates"]] != top10:
            raise ToolError("Top-10 differs from the frozen review; a live multi-agent review is not enabled")
        tiers = {c["name"]: c["tier"] for c in frozen["candidates"]}
        return {"tiers": tiers, "summary": {"source": "frozen multi_agent_review_v1 (reused)",
                                            "tier_counts": frozen["tier_counts"],
                                            "rejected_citations": frozen["citation_validation"]["rejected_quoted_items"]}}

    def report(a: dict) -> dict:
        body = {"top10": a["ranking"]["summary"]["top10"],
                "pathways": a.get("pathways", {}).get("summary", "not computed"),
                "audit": a.get("audit", {}).get("summary", "not computed"),
                "evidence": a.get("evidence", {}).get("tiers", "literature review not run"),
                "boundary": "transcriptomic research hypotheses only; not treatment advice"}
        path = output / "candidate_report.json"
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"summary": {"report": str(path), "evidence_included": "evidence" in a}}

    def transcriptome(_: dict) -> dict:
        from .workflow import run_expression_workflow
        out = output / "expression_ranking"
        result = run_expression_workflow(TRANSCRIPT / "items.csv", TRANSCRIPT / "users.csv", out, mode, 100)
        return {"summary": {"status": result["status"], "manifest": str(out / "manifest.json")}}

    table = {"qc_disease_cohort": qc, "differential_expression": deg, "pathway_enrichment": pathways,
             "rank_candidates": rank, "audit_candidates": audit, "review_literature": review,
             "build_report": report, "rank_transcriptome": transcriptome}

    def run(step: Step, artefacts: dict) -> dict:
        trace.emit("tool_call", tool=step.tool)
        return table[step.tool](artefacts)
    return run
