"""Pathway-level reversal of the LUAD disease signature by the A549 Top-10 (descriptive).

Rules fixed before computing any drug score:
* pathways: MSigDB Hallmark 2020 terms with FDR < 0.05 in
  benchmark/results/luad_pathway_enrichment_v1.json, per direction;
* genes: pathway members that are in the disease signature of that direction
  (FDR < 0.05, |log2FC| >= 1) and among the matched LINCS landmark genes;
  pathways with fewer than 5 such genes are skipped;
* score: reversal = -direction x mean drug signature value over those genes
  (direction +1 for tumour-up pathways, -1 for tumour-down pathways), so a
  positive value means the drug pushes the pathway genes back towards normal;
* reference: all 4,920 A549 drug names; each Top-10 drug gets its percentile
  (1.0 = strongest reversal) per pathway.
The ranking itself is untouched; this only annotates the frozen Top-10.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run

H5 = Path("data/raw/GSE92742/lincs_EH3226.h5")
MIN_GENES = 5


def pathway_reversal(drugs: pd.DataFrame, deg: pd.DataFrame, library: dict[str, list[str]],
                     terms: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    """drugs: genes x drugs. deg: indexed by gene with log2FC, fdr. terms: {'up': [...], 'down': [...]}."""
    sig_up = set(deg.index[(deg.fdr < 0.05) & (deg.log2FC >= 1)])
    sig_down = set(deg.index[(deg.fdr < 0.05) & (deg.log2FC <= -1)])
    genes_available = set(drugs.index)
    scores, used = {}, {}
    for direction, sign, signature in (("up", 1, sig_up), ("down", -1, sig_down)):
        for term in terms.get(direction, []):
            genes = sorted(set(library[term]) & signature & genes_available)
            if len(genes) < MIN_GENES:
                continue
            key = f"{term} ({'肿瘤上调' if direction == 'up' else '肿瘤下调'})"
            scores[key] = -sign * drugs.loc[genes].mean(axis=0)
            used[key] = genes
    return pd.DataFrame(scores), used


def _load_drugs(deg: pd.DataFrame, trace: TraceRecorder) -> pd.DataFrame:
    import h5py
    gene_info = pd.read_csv("data/raw/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz", sep="\t")
    landmark = gene_info.loc[gene_info.pr_is_lm == 1].copy()
    landmark["gene_id"] = landmark.pr_gene_id.astype(str)
    with h5py.File(H5, "r") as h5:
        assay = h5["assay"]
        columns = [x.decode().rstrip("\0") for x in h5["colnames"][0]]
        rows = [x.decode().rstrip("\0") for x in h5["rownames"][0]]
        selected = [(i, v.removesuffix("__A549__trt_cp")) for i, v in enumerate(columns)
                    if v.endswith("__A549__trt_cp")]
        row_map = {g: i for i, g in enumerate(rows)}
        matched = landmark.loc[landmark.gene_id.isin(row_map) & landmark.pr_gene_symbol.isin(deg.index)].copy()
        matched["h5_column"] = [row_map[g] for g in matched.gene_id]
        matched = matched.sort_values("h5_column")
        cols = matched.h5_column.to_numpy()
        values = np.empty((len(matched), len(selected)))
        for j, (i, _) in enumerate(selected):
            values[:, j] = assay[i, cols]
    trace.emit("drugs_loaded", drugs=len(selected), landmark_genes=len(matched))
    return pd.DataFrame(values, index=matched.pr_gene_symbol.astype(str).tolist(),
                        columns=[n for _, n in selected])


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/luad_pathway_reversal_v1.json"))
    args = parser.parse_args()
    if H5.stat().st_size != 2464124175:
        raise ValueError("EH3226 archive incomplete or unexpected")
    deg = pd.read_csv("data/processed/luad/luad_deg_all.tsv", sep="\t").set_index("gene_symbol")
    enrich = json.loads(Path("benchmark/results/luad_pathway_enrichment_v1.json").read_text(encoding="utf-8"))
    hall = enrich["libraries"]["MSigDB_Hallmark_2020"]
    terms = {d: [r["term"] for r in hall[d] if r.get("fdr", 1) < 0.05] for d in ("up", "down")}
    library = json.loads(Path("artifacts/pathway/MSigDB_Hallmark_2020.json").read_text(encoding="utf-8"))
    ranking = pd.read_csv("artifacts/reports/luad_eh3226/all_candidates.csv")
    manifest = json.loads(Path("data/manifests/eh3226-luad.json").read_text(encoding="utf-8"))
    # Input check (amended before any pathway score was computed): the regenerated
    # all_candidates.csv can differ byte-wise from the frozen file because float
    # formatting depends on the NumPy version. Require instead that the frozen Top-10
    # names and the byte-exact positive-control ranks are reproduced, and record
    # whether the full-file hash matched.
    digest = sha256_file(Path("artifacts/reports/luad_eh3226/all_candidates.csv"))
    controls_digest = sha256_file(Path("artifacts/reports/luad_eh3226/positive_control_ranks.csv"))
    if controls_digest != manifest["outputs"]["positive_control_ranks.csv"]:
        raise ValueError("Regenerated positive-control ranks differ from the frozen manifest")
    top10 = ranking.sort_values("rank").drug_name.head(10).tolist()
    frozen = [c["name"] for c in json.loads(Path("configs/luad_top10_evidence_v1.json").read_text(encoding="utf-8"))["candidates"]]
    if top10 != frozen:
        raise ValueError("Regenerated Top-10 differs from the frozen evidence config")
    trace.emit("ranking_verified", top10_matches=True, controls_sha256_matches=True,
               all_candidates_sha256_matches=digest == manifest["outputs"]["all_candidates.csv"])
    drugs = _load_drugs(deg, trace)
    scores, used = pathway_reversal(drugs, deg, library, terms)
    pct = scores.rank(pct=True)
    result = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "rules": __doc__.split("Rules fixed before computing any drug score:")[1].strip(),
              "all_candidates_sha256": digest,
              "all_candidates_sha256_matches_frozen": digest == manifest["outputs"]["all_candidates.csv"],
              "input_check": "frozen Top-10 names and byte-exact positive-control ranks reproduced",
              "top10": top10,
              "pathways": {k: {"genes": v, "n_genes": len(v)} for k, v in used.items()},
              "skipped_pathways": sorted(set(terms["up"]) | set(terms["down"]) -
                                         {k.rsplit(" (", 1)[0] for k in used}),
              "top10_percentile": {d: {k: float(pct.loc[d, k]) for k in pct.columns} for d in top10},
              "top10_score": {d: {k: float(scores.loc[d, k]) for k in scores.columns} for d in top10},
              "top10_mean_percentile_by_pathway": {k: float(pct.loc[top10, k].mean()) for k in pct.columns},
              "trace_file": str(trace.path)}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))
    for k, v in sorted(result["top10_mean_percentile_by_pathway"].items(), key=lambda x: -x[1]):
        print(f"{v:.2f}  n={len(used[k]):3d}  {k}")


def main() -> None:
    traced_run("luad_pathway_reversal_v1", _main)


if __name__ == "__main__":
    main()
