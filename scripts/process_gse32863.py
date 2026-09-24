"""Build a paired LUAD disease signature from GEO GSE32863 and GPL6884."""

from __future__ import annotations

import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from drug_repurposing_agent.differential_expression import paired_deg


ROOT = Path("data/raw/GSE32863")
SERIES = ROOT / "GSE32863_series_matrix.txt.gz"
ANNOTATION = ROOT / "GPL6884.annot.gz"
OUTPUT = Path("data/processed/luad")


def file_hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def table_start(path: Path, marker: str) -> tuple[int, dict[str, list[str]]]:
    metadata = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream):
            if line.startswith(marker):
                return line_number + 1, metadata
            if line.startswith("!Sample_title") or line.startswith("!Sample_geo_accession"):
                parts = next(csv.reader([line], delimiter="\t"))
                metadata[parts[0]] = parts[1:]
    raise ValueError(f"Missing {marker} in {path}")


def sample_manifest(metadata: dict[str, list[str]]) -> pd.DataFrame:
    titles = metadata["!Sample_title"]
    accessions = metadata["!Sample_geo_accession"]
    if len(titles) != 116 or len(accessions) != 116:
        raise ValueError(f"Expected 116 expression samples, got {len(titles)}")
    rows = []
    for accession, title in zip(accessions, titles):
        match = re.search(r"([A-Za-z0-9]+)_([TN]) \(expression\)$", title)
        if not match:
            raise ValueError(f"Cannot parse paired sample title: {title}")
        rows.append({"sample_id": accession, "title": title,
                     "patient_id": match.group(1),
                     "condition": "Tumor" if match.group(2) == "T" else "Normal"})
    manifest = pd.DataFrame(rows)
    pairs = manifest.groupby(["patient_id", "condition"]).size().unstack(fill_value=0)
    complete = pairs.index[(pairs.Tumor == 1) & (pairs.Normal == 1)]
    if len(complete) != 57:
        raise ValueError(f"Expected 57 verifiable pairs from current GEO metadata, got {len(complete)}")
    manifest["paired"] = manifest.patient_id.isin(complete)
    manifest["included"] = manifest.paired
    manifest["reason"] = np.where(manifest.paired, "complete_pair", "unpaired_sample_title")
    return manifest


def main() -> None:
    series_start, metadata = table_start(SERIES, "!series_matrix_table_begin")
    samples = sample_manifest(metadata)
    matrix = pd.read_csv(SERIES, sep="\t", skiprows=series_start, nrows=48803,
                         index_col=0, compression="gzip")
    if matrix.shape != (48803, 116) or set(matrix.columns) != set(samples.sample_id):
        raise ValueError(f"Unexpected expression shape/IDs: {matrix.shape}")
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("GSE32863 expression contains missing or infinite values")
    annotation_start, _ = table_start(ANNOTATION, "!platform_table_begin")
    annotation = pd.read_csv(ANNOTATION, sep="\t", skiprows=annotation_start,
                             compression="gzip", usecols=["ID", "Gene symbol"],
                             dtype=str, comment="!")
    annotation = annotation.drop_duplicates("ID").set_index("ID")
    mapped = matrix.join(annotation, how="left")
    valid = mapped["Gene symbol"].fillna("").str.fullmatch(r"[A-Za-z][A-Za-z0-9.\-]*")
    mapped = mapped.loc[valid].copy()
    if mapped.empty:
        raise ValueError("No unambiguous gene symbols were mapped")
    # One probe per gene, chosen without using tumor/normal labels.
    values = mapped[matrix.columns]
    iqr = values.quantile(0.75, axis=1) - values.quantile(0.25, axis=1)
    mapped["probe_iqr"] = iqr
    mapped["probe_id"] = mapped.index
    selected = (mapped.sort_values(["Gene symbol", "probe_iqr", "probe_id"],
                                   ascending=[True, False, True])
                .drop_duplicates("Gene symbol").set_index("Gene symbol"))
    gene_expression = selected[matrix.columns]
    included_ids = samples.loc[samples.included, "sample_id"]
    deg = paired_deg(gene_expression[included_ids], samples.loc[samples.included])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    samples.to_csv(OUTPUT / "luad_samples.tsv", sep="\t", index=False)
    gene_expression[included_ids].to_csv(OUTPUT / "luad_gene_expression.tsv", sep="\t",
                                         index_label="gene_symbol")
    deg.to_csv(OUTPUT / "luad_deg_all.tsv", sep="\t", index=False)
    (OUTPUT / "luad_up.txt").write_text("\n".join(deg.loc[deg.included_default &
                                                            (deg.direction == "up"), "gene_symbol"]) + "\n")
    (OUTPUT / "luad_down.txt").write_text("\n".join(deg.loc[deg.included_default &
                                                              (deg.direction == "down"), "gene_symbol"]) + "\n")
    qc = {"study": "GSE32863", "platform": "GPL6884", "pairs": 57,
          "samples_total": 116, "samples_included": 114,
          "excluded_samples": samples.loc[~samples.included, "sample_id"].tolist(),
          "pairing_note": "GEO titles 3023_T and 3035_N have no matching opposite tissue; excluded.",
          "probes": len(matrix), "mapped_unambiguous_probes": len(mapped),
          "selected_genes": len(gene_expression),
          "default_up": int((deg.included_default & (deg.direction == "up")).sum()),
          "default_down": int((deg.included_default & (deg.direction == "down")).sum()),
          "method": "paired t-test; Benjamini-Hochberg; tumor-minus-normal log2FC",
          "probe_rule": "highest across-all-samples IQR per gene; probe ID tie-break",
          "threshold": "FDR<0.05 and absolute log2FC>=1",
          "source_sha256": {SERIES.name: file_hash(SERIES), ANNOTATION.name: file_hash(ANNOTATION)}}
    (OUTPUT / "luad_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    manifest = {
        "accession": "GSE32863", "platform": "GPL6884",
        "sources": {
            SERIES.name: {"url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE32nnn/GSE32863/matrix/GSE32863_series_matrix.txt.gz",
                          "bytes": SERIES.stat().st_size, "sha256": file_hash(SERIES)},
            ANNOTATION.name: {"url": "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL6nnn/GPL6884/annot/GPL6884.annot.gz",
                              "bytes": ANNOTATION.stat().st_size, "sha256": file_hash(ANNOTATION)}},
        "qc": qc,
        "outputs": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                    for p in sorted(OUTPUT.iterdir()) if p.is_file()},
    }
    destination = Path("data/manifests/gse32863.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))


if __name__ == "__main__":
    main()
