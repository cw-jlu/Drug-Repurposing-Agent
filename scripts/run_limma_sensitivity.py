"""Run verified paired limma and compare it with the exploratory t-test."""

import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from drug_repurposing_agent.data import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rscript", type=Path, default=Path("Rscript"))
    args = parser.parse_args()
    source = json.loads(Path("data/manifests/gse32863.json").read_text(encoding="utf-8"))
    processed = Path("data/processed/luad")
    for name in ("luad_gene_expression.tsv", "luad_samples.tsv", "luad_deg_all.tsv"):
        if sha256_file(processed / name) != source["outputs"][name]["sha256"]:
            raise ValueError(f"GSE32863 source output changed: {name}")
    out = processed / "limma_sensitivity"
    subprocess.run([str(args.rscript), "scripts/paired_limma_gse32863.R",
                    str(processed / "luad_gene_expression.tsv"),
                    str(processed / "luad_samples.tsv"), str(out)], check=True)
    original = pd.read_csv(processed / "luad_deg_all.tsv", sep="\t").set_index("gene_symbol")
    limma = pd.read_csv(out / "luad_limma_all.tsv", sep="\t").set_index("gene_symbol")
    if set(original.index) != set(limma.index) or len(limma) != 19404:
        raise ValueError("limma gene IDs diverged")
    limma = limma.loc[original.index]
    if not np.isfinite(limma[["log2FC", "p_value", "fdr"]]).all().all():
        raise ValueError("Non-finite limma result")
    effect_delta = float(np.max(np.abs(original.log2FC - limma.log2FC)))
    if effect_delta > 1e-8:
        raise ValueError(f"Paired design effect mismatch: {effect_delta}")
    old_up = set(original.index[(original.fdr < 0.05) & (original.log2FC >= 1)])
    old_down = set(original.index[(original.fdr < 0.05) & (original.log2FC <= -1)])
    new_up = set(limma.index[(limma.fdr < 0.05) & (limma.log2FC >= 1)])
    new_down = set(limma.index[(limma.fdr < 0.05) & (limma.log2FC <= -1)])
    overlap = lambda a, b: len(a & b) / len(a | b) if a | b else 1.0
    session = (out / "limma_session_info.txt").read_text(encoding="utf-8")
    r_match = re.search(r"R version ([0-9.]+)", session)
    limma_match = re.search(r"limma_([0-9.]+)", session)
    if not r_match or not limma_match:
        raise ValueError("R/limma versions missing from session info")
    summary = {"method": "limma lmFit + eBayes, patient fixed effect, BH FDR",
               "r_version": r_match.group(1), "limma_version": limma_match.group(1),
               "source_manifest_sha256": sha256_file(Path("data/manifests/gse32863.json")),
               "outputs": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                           for p in out.iterdir() if p.is_file()},
               "comparison": {"max_abs_log2FC_difference": effect_delta,
                              "max_abs_p_value_difference": float(np.max(np.abs(
                                  original.p_value - limma.p_value))),
                              "max_abs_fdr_difference": float(np.max(np.abs(
                                  original.fdr - limma.fdr))),
                              "paired_t_up": len(old_up), "paired_t_down": len(old_down),
                              "limma_up": len(new_up), "limma_down": len(new_down),
                              "up_jaccard": overlap(old_up, new_up),
                              "down_jaccard": overlap(old_down, new_down)}}
    Path("data/manifests/gse32863-limma.json").write_text(json.dumps(summary, indent=2),
                                                           encoding="utf-8")
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()
