"""Rank LUAD candidates using the frozen ExperimentHub EH3226 A549 subset.

This produces transcriptomic research hypotheses only. It does not infer
clinical efficacy, and unresolved drug identities remain unresolved.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from drug_repurposing_agent.data import ExpressionData
from drug_repurposing_agent.evidence import CandidateLedger
from drug_repurposing_agent.ranking import _rrf, connectivity_gene_sets, score_expressions


H5 = Path("data/raw/GSE92742/lincs_EH3226.h5")
SOURCE_URL = "https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/signatureSearchData/v0.1/lincs.h5"
OUTPUT = Path("artifacts/reports/luad_eh3226")


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if H5.stat().st_size != 2464124175:
        raise ValueError("EH3226 archive has unexpected size")
    deg_path = Path("data/processed/luad/luad_deg_all.tsv")
    gene_path = Path("data/raw/GSE92742/GSE92742_Broad_LINCS_gene_info.txt.gz")
    sig_path = Path("data/raw/GSE92742/GSE92742_Broad_LINCS_sig_info.txt.gz")
    pert_path = Path("data/raw/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz")
    deg = pd.read_csv(deg_path, sep="\t").set_index("gene_symbol")
    gene_info = pd.read_csv(gene_path, sep="\t")
    landmark = gene_info.loc[gene_info.pr_is_lm == 1].copy()
    if len(landmark) != 978 or landmark.pr_gene_symbol.duplicated().any():
        raise ValueError("Unexpected LINCS landmark gene annotation")
    landmark["gene_id"] = landmark.pr_gene_id.astype(str)
    with h5py.File(H5, "r") as h5:
        assay = h5["assay"]
        columns = [x.decode().rstrip("\0") for x in h5["colnames"][0]]
        rows = [x.decode().rstrip("\0") for x in h5["rownames"][0]]
        if assay.shape != (45956, 12328) or len(columns) != assay.shape[0]:
            raise ValueError(f"Unexpected EH3226 assay shape {assay.shape}")
        selected = [(i, value.removesuffix("__A549__trt_cp"))
                    for i, value in enumerate(columns) if value.endswith("__A549__trt_cp")]
        if len(selected) != 4920:
            raise ValueError(f"Unexpected A549 signature count: {len(selected)}")
        names = [name for _, name in selected]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate drug names in derived A549 cohort")
        row_map = {gene_id: i for i, gene_id in enumerate(rows)}
        matched = landmark.loc[landmark.gene_id.isin(row_map) &
                               landmark.pr_gene_symbol.isin(deg.index)].copy()
        if len(matched) < 900:
            raise ValueError("Insufficient LUAD-LINCS landmark gene overlap")
        matched["h5_column"] = [row_map[gene_id] for gene_id in matched.gene_id]
        matched = matched.sort_values("h5_column")
        gene_columns = matched.h5_column.to_numpy()
        # HDF5 is stored as signatures x genes with one compressed signature
        # per chunk. Read only the prespecified A549 rows and landmark columns.
        values = np.empty((len(matched), len(selected)), dtype=float)
        for j, (signature_index, _) in enumerate(selected):
            values[:, j] = assay[signature_index, gene_columns]
    symbols = matched.pr_gene_symbol.astype(str).tolist()
    drug_matrix = pd.DataFrame(values, index=symbols, columns=names)
    disease_matrix = pd.DataFrame({"LUAD": deg.loc[symbols, "log2FC"].to_numpy()}, index=symbols)
    data = ExpressionData(drug_matrix, disease_matrix)
    reversal = score_expressions(data)["spearman_reversal"]["LUAD"]
    up = set(deg.index[(deg.included_default) & (deg.direction == "up")]) & set(symbols)
    down = set(deg.index[(deg.included_default) & (deg.direction == "down")]) & set(symbols)
    connectivity = connectivity_gene_sets(drug_matrix, up, down)
    fusion = _rrf([reversal.to_numpy()[:, None], connectivity.to_numpy()[:, None]])[:, 0]
    sig = pd.read_csv(sig_path, sep="\t", low_memory=False)
    sig = sig.loc[(sig.cell_id == "A549") & (sig.pert_type == "trt_cp") &
                  (sig.pert_itime == "24 h") & (sig.pert_idose == "10 µM")]
    ids = sig.groupby("pert_iname").pert_id.agg(lambda x: sorted(set(x)))
    if set(names) != set(ids.index):
        raise ValueError("EH3226 drug names do not match frozen GEO metadata")
    perts = pd.read_csv(pert_path, sep="\t", low_memory=False).drop_duplicates("pert_id").set_index("pert_id")
    records = []
    for i, name in enumerate(names):
        matching_ids = ids[name]
        drug_id = matching_ids[0] if len(matching_ids) == 1 else None
        metadata = perts.loc[drug_id] if drug_id in perts.index else None
        records.append({"drug_name": name, "pert_id": drug_id,
                        "id_status": "unique" if drug_id else "ambiguous_name",
                        "pubchem_cid": str(metadata.pubchem_cid) if metadata is not None and
                                         pd.notna(metadata.pubchem_cid) else None,
                        "spearman_reversal": float(reversal[name]),
                        "connectivity": float(connectivity[name]),
                        "rrf": float(fusion[i])})
    ranking = pd.DataFrame(records).sort_values(["rrf", "drug_name"], ascending=[False, True])
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    controls = pd.read_csv("configs/luad_positive_controls.csv")
    controls["rank"] = controls.drug_name.map(ranking.set_index("drug_name")["rank"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(OUTPUT / "all_candidates.csv", index=False)
    controls.to_csv(OUTPUT / "positive_control_ranks.csv", index=False)
    run_id = "luad-eh3226-landmark-v1"
    ledger_dir = OUTPUT / "evidence_ledger"
    for record in ranking.head(10).itertuples():
        ledger = CandidateLedger(
            drug_id=record.pert_id or record.drug_name, disease_id="LUAD",
            trace_id=run_id, transcriptomic_rank=int(record.rank),
            reversal_scores={"spearman_reversal": record.spearman_reversal,
                             "connectivity": record.connectivity, "rrf": record.rrf},
            drug_name=record.drug_name,
            perturbation_context={"cell_line": "A549", "dose": "10 µM", "time": "24 h",
                                  "source": "ExperimentHub EH3226"},
            limitations=["In vitro A549 signature is not clinical efficacy.",
                         "No target, safety, or literature evidence has been curated."] +
                        (["Drug name maps to multiple GEO perturbagen IDs."]
                         if record.id_status != "unique" else []))
        ledger.save(ledger_dir / f"rank_{record.rank:02d}.json")
    manifest = {"source": "ExperimentHub EH3226 derived from GEO GSE92742",
                "source_url": SOURCE_URL, "source_bytes": H5.stat().st_size,
                "source_sha256": file_hash(H5), "assay_shape": [45956, 12328],
                "cohort": "A549; trt_cp; 10 µM; 24 h; first technical duplicate retained by upstream sig_filter",
                "a549_names": len(ranking), "ambiguous_id_names": int((ranking.id_status != "unique").sum()),
                "landmark_genes_intersected": len(symbols),
                "significant_up_landmark": len(up), "significant_down_landmark": len(down),
                "disease_input_sha256": file_hash(deg_path),
                "method": "negative Spearman + prespecified disease up/down connectivity; RRF k=60",
                "status": "transcriptomic_candidates_only; evidence curation pending",
                "outputs": {p.name: file_hash(p) for p in (OUTPUT / "all_candidates.csv",
                                                           OUTPUT / "positive_control_ranks.csv")}}
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"a549_names": manifest["a549_names"], "landmark_genes_intersected": len(symbols),
                      "up": len(up), "down": len(down), "ambiguous_id_names": manifest["ambiguous_id_names"],
                      "top10": ranking.head(10)[["rank", "drug_name", "pert_id", "rrf"]].to_dict("records")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
