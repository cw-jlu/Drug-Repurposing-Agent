"""Audit top-ranked name mappings against GEO and Broad annotations.

The EH3226 matrix retains drug names, not perturbagen IDs. A match here is
an annotation audit, not proof that the underlying signature used that lot.
"""

from pathlib import Path
import json

import pandas as pd

from drug_repurposing_agent.data import sha256_file


ROOT = Path("artifacts/reports/luad_eh3226")
HUB = Path("data/raw/repurposing_hub")


def norm(value: object) -> str | None:
    if pd.isna(value) or str(value).strip() in {"", "-666"}:
        return None
    return str(value).strip()


def main() -> None:
    ranking = pd.read_csv(ROOT / "all_candidates.csv").head(10)
    geo = pd.read_csv("data/raw/GSE92742/GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t")
    drug = pd.read_csv(HUB / "repo-drug-annotation-20250818.txt", sep="\t", skiprows=9)
    sample = pd.read_csv(HUB / "repo-sample-annotation-20250818.txt", sep="\t", skiprows=9,
                         low_memory=False)
    rows = []
    for candidate in ranking.itertuples():
        g = geo.loc[geo.pert_id == candidate.pert_id] if pd.notna(candidate.pert_id) else geo.iloc[0:0]
        d = drug.loc[drug.pert_iname.str.casefold() == candidate.drug_name.casefold()]
        s = sample.loc[sample.pert_iname.str.casefold() == candidate.drug_name.casefold()]
        geo_key = norm(g.inchi_key.iloc[0]) if len(g) == 1 else None
        geo_cid = norm(g.pubchem_cid.iloc[0]) if len(g) == 1 else None
        hub_keys = {norm(v) for v in s.InChIKey if norm(v)}
        hub_cids = {str(int(v)) for v in s.pubchem_cid if pd.notna(v)}
        hub_ids = {str(v).split("-001-")[0] for v in s.broad_id if pd.notna(v)}
        if candidate.id_status != "unique":
            status = "ambiguous_GEO_name"
        elif not len(s):
            status = "no_Hub_sample"
        elif geo_key and geo_key in hub_keys:
            status = "exact_InChIKey"
        elif geo_cid and geo_cid in hub_cids:
            status = "exact_PubChem_CID"
        elif candidate.pert_id in hub_ids:
            status = "matching_Broad_ID_only"
        elif geo_key and geo_key[:14] in {key[:14] for key in hub_keys}:
            status = "same_connectivity_different_stereochemistry"
        else:
            status = "identity_mismatch"
        rows.append({
            "rank": candidate.rank, "drug_name": candidate.drug_name,
            "geo_pert_id": candidate.pert_id if pd.notna(candidate.pert_id) else None,
            "geo_inchi_key": geo_key, "geo_pubchem_cid": geo_cid,
            "hub_sample_count": len(s), "hub_identity_status": status,
            "hub_clinical_phase_name_annotation": " | ".join(sorted(set(d.clinical_phase.dropna()))) or None,
            "hub_moa_name_annotation": " | ".join(sorted(set(d.moa.dropna()))) or None,
            "hub_targets_name_annotation": " | ".join(sorted(set(d.target.dropna()))) or None,
        })
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "identity_audit.csv", index=False)
    source_manifest = json.loads(Path("data/manifests/repurposing-hub-2025-08-18.json").read_text(
        encoding="utf-8"))
    for name, entry in source_manifest.items():
        if sha256_file(HUB / name) != entry["sha256"]:
            raise ValueError(f"Hub source checksum mismatch: {name}")
    (ROOT / "identity_audit_manifest.json").write_text(json.dumps({
        "source": "Broad Drug Repurposing Hub annotations, 2025-08-18",
        "drug_url": "https://repo-hub.broadinstitute.org/public/data/repo-drug-annotation-20250818.txt",
        "sample_url": "https://repo-hub.broadinstitute.org/public/data/repo-sample-annotation-20250818.txt",
        "rule": "name annotation only; chemical identity requires exact InChIKey or PubChem CID",
        "source_sha256": {name: entry["sha256"] for name, entry in source_manifest.items()},
        "identity_audit_sha256": sha256_file(ROOT / "identity_audit.csv"),
        "counts": output.hub_identity_status.value_counts().to_dict(),
    }, indent=2), encoding="utf-8")
    print(output[["rank", "drug_name", "hub_identity_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
