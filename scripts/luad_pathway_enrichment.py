"""Over-representation analysis of the GSE32863 LUAD disease signature.

Up- and down-regulated genes (FDR < 0.05, |log2FC| >= 1, the prespecified
signature) are tested separately against MSigDB Hallmark 2020 and KEGG 2021
Human (Enrichr libraries). The background is every tested gene that occurs in
the library. One-sided hypergeometric test, Benjamini-Hochberg per library and
direction. Gene-set files are cached under ignored artifacts/pathway and their
SHA-256 recorded in data/manifests/enrichr-gene-sets.json.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run

LIBRARIES = ("MSigDB_Hallmark_2020", "KEGG_2021_Human")
CACHE = Path("artifacts/pathway")
MANIFEST = Path("data/manifests/enrichr-gene-sets.json")


def load_library(name: str, trace: TraceRecorder) -> dict[str, list[str]]:
    path = CACHE / f"{name}.json"
    if not path.exists():
        import gseapy
        CACHE.mkdir(parents=True, exist_ok=True)
        library = gseapy.get_library(name=name, organism="Human")
        path.write_text(json.dumps(library, sort_keys=True), encoding="utf-8")
        trace.emit("library_downloaded", library=name)
    digest = sha256_file(path)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    recorded = manifest.get(name, {}).get("sha256")
    if recorded and recorded != digest:
        raise ValueError(f"{name} differs from the recorded manifest hash")
    trace.emit("library_loaded", library=name, sha256=digest)
    return json.loads(path.read_text(encoding="utf-8"))


def ora(selected: set[str], background: set[str], library: dict[str, list[str]]) -> list[dict]:
    rows = []
    n_bg, n_sel = len(background), len(selected & background)
    for term, genes in library.items():
        members = set(genes) & background
        overlap = sorted(members & selected)
        if len(members) < 10:
            continue
        k = len(overlap)
        p = float(hypergeom.sf(k - 1, n_bg, len(members), n_sel)) if k else 1.0
        expected = n_sel * len(members) / n_bg
        rows.append({"term": term, "set_size": len(members), "overlap": k,
                     "expected": round(expected, 2),
                     "fold_enrichment": round(k / expected, 3) if expected else None,
                     "p_value": p, "genes": overlap[:25]})
    if rows:
        fdr = multipletests([r["p_value"] for r in rows], method="fdr_bh")[1]
        for r, q in zip(rows, fdr):
            r["fdr"] = float(q)
    return sorted(rows, key=lambda r: (r["p_value"], r["term"]))


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/luad_pathway_enrichment_v1.json"))
    args = parser.parse_args()
    from scripts.make_figures import load_deg
    deg, source = load_deg(trace)
    deg = deg.dropna(subset=["log2FC", "fdr"])
    up = set(deg.loc[(deg.fdr < 0.05) & (deg.log2FC >= 1), "gene_symbol"])
    down = set(deg.loc[(deg.fdr < 0.05) & (deg.log2FC <= -1), "gene_symbol"])
    tested = set(deg.gene_symbol)
    trace.emit("signature", up=len(up), down=len(down), tested=len(tested), source=source)
    manifest = {}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "deg_source": source,
              "thresholds": "FDR < 0.05 and |log2FC| >= 1 (prespecified)",
              "test": "one-sided hypergeometric; BH within library x direction; sets with < 10 background genes skipped",
              "signature_sizes": {"up": len(up), "down": len(down), "tested": len(tested)},
              "libraries": {}, "trace_file": str(trace.path)}
    for name in LIBRARIES:
        library = load_library(name, trace)
        manifest[name] = {"source": "Enrichr via gseapy.get_library", "terms": len(library),
                          "sha256": sha256_file(CACHE / f"{name}.json"),
                          "retrieved": datetime.now(timezone.utc).date().isoformat()}
        background = tested & set().union(*map(set, library.values()))
        report["libraries"][name] = {
            "background_genes": len(background),
            "up": ora(up, background, library), "down": ora(down, background, library)}
        for direction in ("up", "down"):
            sig = [r["term"] for r in report["libraries"][name][direction] if r.get("fdr", 1) < 0.05]
            trace.emit("enrichment", library=name, direction=direction, significant=len(sig), top=sig[:5])
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))
    for name, body in report["libraries"].items():
        for direction in ("up", "down"):
            top = [r for r in body[direction] if r.get("fdr", 1) < 0.05][:8]
            print(f"{name} {direction}: {len([r for r in body[direction] if r.get('fdr',1)<0.05])} FDR<0.05")
            for r in top:
                print(f"   {r['term'][:48]:48s} k={r['overlap']:3d}/{r['set_size']:3d} FE={r['fold_enrichment']:.2f} FDR={r['fdr']:.1e}")


def main() -> None:
    traced_run("luad_pathway_enrichment_v1", _main)


if __name__ == "__main__":
    main()
