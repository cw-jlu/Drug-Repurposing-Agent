"""Frozen public-input construction for the LUAD evidence-decision holdout."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file


DEFAULT_CASES = Path("configs/agent_v4_claim_holdout.json")


def load_holdout(path: Path = DEFAULT_CASES) -> tuple[dict, dict[int, dict]]:
    holdout = json.loads(path.read_text(encoding="utf-8"))
    if holdout.get("name") != "agent_v4_luad_evidence_decision_holdout":
        raise ValueError("Unexpected holdout identity")
    for source, expected in holdout["source_sha256"].items():
        if sha256_file(Path(source)) != expected:
            raise ValueError(f"Frozen holdout source differs: {source}")
    cases = holdout["cases"]
    if len(cases) != 20 or len({case["id"] for case in cases}) != 20:
        raise ValueError("Expected 20 unique frozen cases")
    with Path("configs/luad_top10_signature_ids.csv").open(encoding="utf-8-sig", newline="") as handle:
        signatures = {int(row["rank"]): row for row in csv.DictReader(handle)}
    evidence = json.loads(Path("configs/luad_top10_evidence_v1.json").read_text(encoding="utf-8"))
    names = {candidate["rank"]: candidate["name"] for candidate in evidence["candidates"]}
    if set(signatures) != set(names):
        raise ValueError("LUAD signature and evidence ranks differ")
    for rank, row in signatures.items():
        if row["drug_name"] != names[rank]:
            raise ValueError(f"Crosswalk name differs at rank {rank}")
    return holdout, signatures


def public_case(case: dict, signatures: dict[int, dict]) -> dict:
    source = signatures[case["rank"]]
    return {"case_id": case["id"], "drug_name": source["drug_name"],
            "source_sig_id": source["sig_id"], "source_pert_id": source["pert_id"],
            "source_inchi_key": source["inchi_key"],
            "cross_source_identity_status": case["identity"],
            "request": case["request"], "evidence_cards": case["cards"]}
