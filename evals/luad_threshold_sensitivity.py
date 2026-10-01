"""Sensitivity of the frozen LUAD A549 Top-10 to the disease-signature thresholds.

The frozen ranking fuses (RRF, k=60) the negative Spearman reversal over all matched
landmark genes, which does not depend on any threshold, with up/down gene-set
connectivity, which does. Here the connectivity sets are rebuilt under:
* threshold grid: FDR in {0.01, 0.05, 0.10} x |log2FC| in {0.58, 1.0, 1.5};
* top-N variants: the N landmark genes with the largest |log2FC| per direction
  among FDR < 0.05 genes, N in {25, 50, 100}.
The default (FDR 0.05, |log2FC| 1) must reproduce the frozen Top-10 exactly.
Per variant we report the Top-10, the overlap with the frozen Top-10 (count and
Jaccard), the number of corticosteroids in the Top-10 (name rule below, printed
for audit) and the mean rank percentile of the measured reference drugs.
This is a robustness check; the frozen ranking and evidence are not changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from drug_repurposing_agent.data import ExpressionData, sha256_file
from drug_repurposing_agent.ranking import _rrf, connectivity_gene_sets, score_expressions
from drug_repurposing_agent.trace import TraceRecorder, traced_run

FDRS = (0.01, 0.05, 0.10)
LFCS = (0.58, 1.0, 1.5)
TOP_NS = (25, 50, 100)
# Corticosteroid name rule (audited by printing every Top-10 name with its label).
STEROID = re.compile(r"(cort|sone|solone|olone|nide|betamethasone|dexamethasone|prednis|"
                     r"triamcinolone|fluocinolone|halcinonide|amcinonide)", re.I)


def is_steroid(name: str) -> bool:
    return bool(STEROID.search(name)) and "cortistatin" not in name.lower()


def rank(drugs: pd.DataFrame, reversal: np.ndarray, up: set[str], down: set[str]) -> pd.Series:
    connectivity = connectivity_gene_sets(drugs, up, down).to_numpy()
    fused = _rrf([reversal[:, None], connectivity[:, None]])[:, 0]
    frame = pd.DataFrame({"drug_name": drugs.columns, "rrf": fused})
    frame = frame.sort_values(["rrf", "drug_name"], ascending=[False, True])
    return pd.Series(np.arange(1, len(frame) + 1), index=frame.drug_name)


def summarize(ranks: pd.Series, frozen: list[str], controls: list[str]) -> dict:
    top = ranks.index[:10].tolist()
    overlap = len(set(top) & set(frozen))
    measured = [c for c in controls if c in ranks.index]
    return {"top10": top, "overlap_with_frozen": overlap,
            "jaccard_with_frozen": round(overlap / len(set(top) | set(frozen)), 3),
            "steroids_in_top10": sum(map(is_steroid, top)),
            "control_mean_percentile": float(np.mean([ranks[c] / len(ranks) for c in measured])) if measured else None}


def _main(trace: TraceRecorder) -> None:
    from scripts.luad_pathway_reversal import _load_drugs
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/luad_threshold_sensitivity_v1.json"))
    args = parser.parse_args()
    deg = pd.read_csv("data/processed/luad/luad_deg_all.tsv", sep="\t").set_index("gene_symbol")
    drugs = _load_drugs(deg, trace)
    symbols = drugs.index.tolist()
    disease = pd.DataFrame({"LUAD": deg.loc[symbols, "log2FC"].to_numpy()}, index=symbols)
    reversal = score_expressions(ExpressionData(drugs, disease))["spearman_reversal"]["LUAD"].to_numpy()
    lm = deg.loc[symbols]
    frozen = [c["name"] for c in json.loads(Path("configs/luad_top10_evidence_v1.json").read_text(encoding="utf-8"))["candidates"]]
    controls = pd.read_csv("configs/luad_positive_controls.csv").drug_name.tolist()
    variants = {}
    for fdr in FDRS:
        for lfc in LFCS:
            up = set(lm.index[(lm.fdr < fdr) & (lm.log2FC >= lfc)])
            down = set(lm.index[(lm.fdr < fdr) & (lm.log2FC <= -lfc)])
            key = f"FDR<{fdr:g}, |log2FC|>={lfc:g}"
            variants[key] = {"up": len(up), "down": len(down), **summarize(rank(drugs, reversal, up, down), frozen, controls)}
    sig = lm.loc[lm.fdr < 0.05]
    for n in TOP_NS:
        up = set(sig.loc[sig.log2FC > 0].log2FC.nlargest(n).index)
        down = set(sig.loc[sig.log2FC < 0].log2FC.nsmallest(n).index)
        variants[f"top-{n} per direction (FDR<0.05)"] = {"up": len(up), "down": len(down),
                                                         **summarize(rank(drugs, reversal, up, down), frozen, controls)}
    default = variants["FDR<0.05, |log2FC|>=1"]
    if default["top10"] != frozen:
        raise ValueError("Default thresholds do not reproduce the frozen Top-10")
    names = sorted({n for v in variants.values() for n in v["top10"]})
    hub_path = Path("data/raw/repurposing_hub/repo-drug-annotation-20250818.txt")
    hub = pd.read_csv(hub_path, sep="	", comment="!").drop_duplicates("pert_iname").set_index("pert_iname")
    moa = {n: (str(hub.loc[n, "moa"]) if n in hub.index and pd.notna(hub.loc[n, "moa"]) else None) for n in names}
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "frozen_top10": frozen,
              "default_reproduces_frozen": True, "landmark_genes": len(symbols),
              "steroid_rule": STEROID.pattern, "steroid_labels": {n: is_steroid(n) for n in names},
              "hub_moa": moa, "hub_annotation_sha256": sha256_file(hub_path),
              "variants": variants,
              "summary": {"min_overlap": min(v["overlap_with_frozen"] for v in variants.values()),
                          "min_steroids": min(v["steroids_in_top10"] for v in variants.values()),
                          "always_in_top10": sorted(set.intersection(*(set(v["top10"]) for v in variants.values())))},
              "deg_sha256": sha256_file(Path("data/processed/luad/luad_deg_all.tsv")), "trace_file": str(trace.path)}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))
    for k, v in variants.items():
        print(f"{k:34s} up={v['up']:3d} down={v['down']:3d} overlap={v['overlap_with_frozen']:2d} steroids={v['steroids_in_top10']:2d} ctrl_pct={v['control_mean_percentile']:.3f}")
    print("labels:", result["steroid_labels"])
    print("moa:", {k: v for k, v in moa.items() if not is_steroid(k)})
    print("summary:", result["summary"])


def main() -> None:
    traced_run("luad_threshold_sensitivity_v1", _main)


if __name__ == "__main__":
    main()
