"""Freeze the GSE92742 A549 chemical signature cohort before any ranking."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


ROOT = Path("data/raw/GSE92742")
PREFIX = "GSE92742_Broad_LINCS_"
OUTPUT = Path("data/processed/luad/a549_signature_manifest.tsv")


def file_hash(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    paths = {name: ROOT / f"{PREFIX}{name}.txt.gz"
             for name in ("sig_info", "gene_info", "pert_info")}
    sig = pd.read_csv(paths["sig_info"], sep="\t", low_memory=False)
    genes = pd.read_csv(paths["gene_info"], sep="\t", low_memory=False)
    perts = pd.read_csv(paths["pert_info"], sep="\t", low_memory=False)
    required = {"sig_id", "pert_id", "pert_iname", "pert_type", "cell_id", "pert_itime", "pert_idose"}
    if not required.issubset(sig.columns):
        raise ValueError(f"Missing signature columns: {required - set(sig.columns)}")
    selected = sig.loc[(sig.cell_id == "A549") & (sig.pert_type == "trt_cp") &
                       (sig.pert_itime == "24 h") & (sig.pert_idose == "10 µM")].copy()
    if selected.sig_id.duplicated().any() or selected.empty:
        raise ValueError("Missing or duplicate selected signatures")
    selected = selected.sort_values("sig_id")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT, sep="\t", index=False)
    controls = pd.read_csv("configs/luad_positive_controls.csv")
    available_names = set(selected.pert_iname.str.lower())
    control_presence = {name: name in available_names for name in controls.drug_name}
    manifest = {
        "accession": "GSE92742", "cohort": "A549 trt_cp 24 h 10 µM",
        "status": "metadata_only; Level 5 GCTX expression not downloaded",
        "sig_info_total": len(sig), "a549_trt_cp_total": int(((sig.cell_id == "A549") &
                                                              (sig.pert_type == "trt_cp")).sum()),
        "selected_signatures": len(selected), "selected_pert_ids": selected.pert_id.nunique(),
        "gene_info_rows": len(genes), "pert_info_rows": len(perts),
        "positive_control_name_presence": control_presence,
        "selection_rule": "cell_id=A549; pert_type=trt_cp; pert_itime=24 h; pert_idose=10 µM; no post-ranking changes",
        "sources": {path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size,
                                "url": f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/{path.name}"}
                    for name, path in paths.items()},
        "selected_manifest_sha256": file_hash(OUTPUT),
    }
    destination = Path("data/manifests/gse92742-a549.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "selected_signatures",
                                                   "selected_pert_ids", "positive_control_name_presence")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
