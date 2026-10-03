"""Feasibility check for a second registry disease: breast cancer, GEO GSE15852 + LINCS MCF7.

Not a registry entry and not a result to report as a finding: it only checks
whether the registry-driven tools (geo_cohort + the existing ranking code) run
unchanged on another disease. Steps: download the series matrix and the GPL96
annotation (hashes recorded so a future registry entry can pin them), pair the
samples with a title rule, recompute the paired signature with automatic log2
detection, Hallmark enrichment, and rank MCF7 compound signatures. No reference
drugs are prespecified for breast cancer, so recovery is not evaluated; the
multi-agent literature reviewer is LUAD-specific and is not run.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import urllib.request

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.geo_cohort import USER_AGENT, cohort, file_sha256, signature
from drug_repurposing_agent.trace import TraceRecorder, traced_run

RAW = Path("data/raw/GSE15852")
FILES = {"series": ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE15nnn/GSE15852/matrix/GSE15852_series_matrix.txt.gz",
                    RAW / "GSE15852_series_matrix.txt.gz"),
         "annotation": ("https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL96/annot/GPL96.annot.gz",
                        RAW / "GPL96.annot.gz")}
OUTPUT = Path("benchmark/results/disease_feasibility_gse15852.json")


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)


def _main(trace: TraceRecorder) -> None:
    timings, files = {}, {}
    t = perf_counter()
    for role, (url, path) in FILES.items():
        if not path.is_file():
            _download(url, path)
        files[role] = {"path": path.as_posix(), "url": url, "bytes": path.stat().st_size,
                       "sha256": file_sha256(path)}
        trace.emit("file_ready", role=role, **files[role])
    timings["download_s"] = round(perf_counter() - t, 1)
    entry = {"id": "breast_gse15852_candidate", "label": "乳腺癌（候选，未登记）", "accession": "GSE15852",
             "platform": "GPL96", "files": files,
             "pairing": {"field": "!Sample_title", "regex": r"(BC\d+)([NT])$", "case": "T", "control": "N"},
             "min_pairs": 10, "values": "auto", "annotation_symbol_column": "Gene symbol",
             "drug_cell_line": "MCF7"}
    t = perf_counter()
    samples = cohort(entry)
    timings["cohort_s"] = round(perf_counter() - t, 1)
    t = perf_counter()
    deg, sig = signature(entry, samples)
    timings["signature_s"] = round(perf_counter() - t, 1)
    deg = deg.dropna(subset=["log2FC", "fdr"])

    from scripts.luad_pathway_enrichment import load_library, ora
    lib = load_library("MSigDB_Hallmark_2020", trace)
    up = set(deg.loc[(deg.fdr < 0.05) & (deg.log2FC >= 1), "gene_symbol"])
    down = set(deg.loc[(deg.fdr < 0.05) & (deg.log2FC <= -1), "gene_symbol"])
    background = set(deg.gene_symbol) & set().union(*map(set, lib.values()))
    top = lambda rows: [r["term"] for r in rows if r.get("fdr", 1) < 0.05][:5]
    pathways = {"up": top(ora(up, background, lib)), "down": top(ora(down, background, lib))}

    import pandas as pd
    from drug_repurposing_agent.data import ExpressionData
    from drug_repurposing_agent.ranking import score_expressions
    from evals.luad_threshold_sensitivity import rank as rank_fn
    from scripts.luad_pathway_reversal import _load_drugs
    t = perf_counter()
    table = deg.set_index("gene_symbol")
    drugs = _load_drugs(table, trace, cell_line="MCF7")
    genes = drugs.index.tolist()
    vec = pd.DataFrame({"disease": table.loc[genes, "log2FC"].to_numpy()}, index=genes)
    reversal = score_expressions(ExpressionData(drugs, vec))["spearman_reversal"]["disease"].to_numpy()
    lm = table.loc[genes]
    ranks = rank_fn(drugs, reversal, set(lm.index[(lm.fdr < 0.05) & (lm.log2FC >= 1)]),
                    set(lm.index[(lm.fdr < 0.05) & (lm.log2FC <= -1)]))
    timings["ranking_s"] = round(perf_counter() - t, 1)

    result = {"eval_name": "disease_feasibility_gse15852", "note": __doc__, "files": files,
              "cohort": {"samples": len(samples), "included": int(samples.included.sum()),
                         "pairs": int(samples.included.sum()) // 2,
                         "excluded": samples.loc[~samples.included, ["sample_id", "title", "reason"]].to_dict("records")},
              "signature": sig, "pathways_hallmark_top5": pathways,
              "ranking": {"cell_line": "MCF7", "compounds": len(ranks), "landmark_genes_matched": len(genes),
                          "signature_genes_on_landmarks": {"up": len(set(lm.index[(lm.fdr < 0.05) & (lm.log2FC >= 1)])),
                                                           "down": len(set(lm.index[(lm.fdr < 0.05) & (lm.log2FC <= -1)]))},
                          "top10": ranks.index[:10].tolist()},
              "timings": timings, "trace_file": str(trace.path)}
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(OUTPUT), sha256=sha256_file(OUTPUT))
    print(json.dumps({k: result[k] for k in ("cohort", "signature", "pathways_hallmark_top5", "ranking", "timings")},
                     ensure_ascii=False, indent=1, default=str)[:3000])


def main() -> None:
    traced_run("disease_feasibility_gse15852", _main)


if __name__ == "__main__":
    main()
