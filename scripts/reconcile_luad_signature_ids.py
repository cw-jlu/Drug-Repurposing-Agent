"""Recover the exact GSE92742 signature and perturbagen IDs used by EH3226.

EH3226 stores ``pert_iname__cell_id__pert_type`` as its column identifier.  Its
published construction recipe filters Level 5 metadata and then keeps the first
row for each ``pert_iname + cell_id`` pair.  Replaying that deterministic step
against the checksum-pinned GEO metadata recovers the source ``sig_id`` without
reading or re-ranking the 2.46 GB expression matrix.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pandas as pd


SIG_INFO = Path("data/raw/GSE92742/GSE92742_Broad_LINCS_sig_info.txt.gz")
PERT_INFO = Path("data/raw/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz")
EVIDENCE = Path("configs/luad_top10_evidence_v1.json")
OUTPUT = Path("configs/luad_top10_signature_ids.csv")
MANIFEST = Path("data/manifests/luad-top10-signatures-v1.json")
UPSTREAM_SOURCE = (
    "https://rdrr.io/github/yduan004/signatureSearch_data/src/R/meta_filter.R"
)


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: object) -> str | None:
    if pd.isna(value) or str(value) in {"-666", "nan"}:
        return None
    return str(value)


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    candidates = pd.DataFrame(
        [{"rank": row["rank"], "drug_name": row["name"]}
         for row in evidence["candidates"]]
    )
    sig = pd.read_csv(SIG_INFO, sep="\t", low_memory=False)
    selected = sig.loc[
        (sig.cell_id == "A549")
        & (sig.pert_type == "trt_cp")
        & (sig.pert_itime == "24 h")
        & (sig.pert_idose == "10 µM")
    ].copy()
    before_distinct = len(selected)
    # This exactly matches dplyr::distinct(alt_id, .keep_all=TRUE) in sig_filter.
    selected = selected.drop_duplicates(["pert_iname", "cell_id"], keep="first")
    if len(selected) != 4920 or selected.pert_iname.duplicated().any():
        raise ValueError("Unexpected reconstructed EH3226 A549 cohort")

    recovered = candidates.merge(
        selected,
        left_on="drug_name",
        right_on="pert_iname",
        how="left",
        validate="one_to_one",
    )
    if recovered.sig_id.isna().any():
        missing = recovered.loc[recovered.sig_id.isna(), "drug_name"].tolist()
        raise ValueError(f"Top-10 names missing from reconstructed cohort: {missing}")

    pert = pd.read_csv(PERT_INFO, sep="\t", low_memory=False)
    pert = pert.drop_duplicates("pert_id", keep="first")
    recovered = recovered.merge(
        pert[["pert_id", "is_touchstone", "inchi_key", "canonical_smiles", "pubchem_cid"]],
        on="pert_id",
        how="left",
        validate="many_to_one",
    )
    columns = [
        "rank", "drug_name", "sig_id", "pert_id", "cell_id", "pert_type",
        "pert_dose", "pert_dose_unit", "pert_idose", "pert_time",
        "pert_time_unit", "pert_itime", "distil_id", "is_touchstone",
        "inchi_key", "canonical_smiles", "pubchem_cid",
    ]
    output = recovered[columns].copy()
    for column in ("inchi_key", "canonical_smiles", "pubchem_cid"):
        output[column] = output[column].map(clean)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)

    manifest = {
        "version": "luad_top10_signatures_v1",
        "status": "exact_signature_ids_recovered_from_pinned_metadata",
        "reconstruction_rule": (
            "filter pert_type=trt_cp, pert_idose=10 µM, pert_itime=24 h; "
            "then keep the first source-order row per pert_iname+cell_id"
        ),
        "upstream_method_source": UPSTREAM_SOURCE,
        "source_filtered_rows_before_distinct": before_distinct,
        "reconstructed_a549_columns": len(selected),
        "top10_recovered": len(output),
        "top10_unique_sig_ids": int(output.sig_id.nunique()),
        "top10_unique_pert_ids": int(output.pert_id.nunique()),
        "source_sha256": {
            str(SIG_INFO): file_hash(SIG_INFO),
            str(PERT_INFO): file_hash(PERT_INFO),
            str(EVIDENCE): file_hash(EVIDENCE),
        },
        "output": {"path": str(OUTPUT), "sha256": file_hash(OUTPUT)},
        "limitations": [
            "The reconstruction identifies the exact source signature but does not repeat the expression ranking.",
            "Chemical structures still require row-level identity review when Broad and Hub annotations disagree.",
            "The signatures are A549 in vitro perturbations and do not establish treatment efficacy.",
        ],
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "top10_recovered": manifest["top10_recovered"],
        "unique_sig_ids": manifest["top10_unique_sig_ids"],
        "unique_pert_ids": manifest["top10_unique_pert_ids"],
        "output_sha256": manifest["output"]["sha256"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
