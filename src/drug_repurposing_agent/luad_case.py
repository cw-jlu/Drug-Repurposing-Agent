"""Validate and package the LUAD expression screen as an auditable case run.

This stage intentionally cannot promote a transcriptomic hit to an evidence-
supported treatment candidate. It consumes frozen outputs and checks their
provenance before assembling a research report and execution trace.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from .data import sha256_file
from .evidence import CandidateLedger, Citation
from .jev import JevClient, choice_question, gate_choice


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"Source hash mismatch: {path}")


def _validated_ledger(path: Path) -> CandidateLedger:
    raw = _read_json(path)
    for name in ("supporting_evidence", "contradicting_evidence", "context_evidence"):
        raw[name] = [Citation(**entry) for entry in raw.get(name, [])]
    ledger = CandidateLedger(**raw)
    ledger.validate()
    return ledger


def build_luad_case(screen_dir: Path, disease_manifest: Path, screen_manifest: Path,
                    output: Path, jev_client: JevClient | None = None) -> dict:
    """Assemble a report only when all upstream rows and hashes agree."""
    started = perf_counter()
    run_id = uuid4().hex
    trace: list[dict] = []

    def step(stage: str, **details: object) -> None:
        trace.append({"time": datetime.now(timezone.utc).isoformat(),
                      "stage": stage, **details})

    step("plan", disease="LUAD", mode="research_open",
         allowed_sources=["frozen_GEO", "frozen_ExperimentHub", "Broad_identity_audit"],
         external_model_calls=0)
    disease = _read_json(disease_manifest)
    screen = _read_json(screen_manifest)
    if disease["qc"]["pairs"] != 57 or screen["a549_names"] != 4920:
        raise ValueError("Unexpected cohort dimensions")
    _check_hash(Path("data/raw/GSE92742/lincs_EH3226.h5"), screen["source_sha256"])
    for name, entry in disease["outputs"].items():
        _check_hash(Path("data/processed/luad") / name, entry["sha256"])
    _check_hash(Path("data/processed/luad/luad_deg_all.tsv"),
                screen["disease_input_sha256"])
    for name, expected in screen["outputs"].items():
        _check_hash(screen_dir / name, expected)
    limma_manifest_path = Path("data/manifests/gse32863-limma.json")
    sensitivity = None
    if limma_manifest_path.exists():
        limma_manifest = _read_json(limma_manifest_path)
        if limma_manifest["source_manifest_sha256"] != sha256_file(disease_manifest):
            raise ValueError("limma sensitivity source manifest differs")
        for name, entry in limma_manifest["outputs"].items():
            _check_hash(Path("data/processed/luad/limma_sensitivity") / name,
                        entry["sha256"])
        sensitivity = {"method": limma_manifest["method"],
                       "r_version": limma_manifest["r_version"],
                       "limma_version": limma_manifest["limma_version"],
                       "comparison": limma_manifest["comparison"]}
    identity_manifest = _read_json(screen_dir / "identity_audit_manifest.json")
    _check_hash(screen_dir / "identity_audit.csv", identity_manifest["identity_audit_sha256"])
    for name, expected in identity_manifest["source_sha256"].items():
        _check_hash(Path("data/raw/repurposing_hub") / name, expected)
    step("source_verified", disease_pairs=57, a549_names=4920,
         source_sha256=screen["source_sha256"])

    ranking = pd.read_csv(screen_dir / "all_candidates.csv")
    identity = pd.read_csv(screen_dir / "identity_audit.csv")
    controls = pd.read_csv(screen_dir / "positive_control_ranks.csv")
    if len(ranking) != screen["a549_names"] or ranking.drug_name.duplicated().any():
        raise ValueError("Candidate count or name uniqueness mismatch")
    if not np.array_equal(ranking["rank"].to_numpy(), np.arange(1, len(ranking) + 1)):
        raise ValueError("Ranks are not consecutive")
    if not np.isfinite(ranking[["spearman_reversal", "connectivity", "rrf"]]).all().all():
        raise ValueError("Non-finite candidate score")
    if (ranking.rrf.diff().iloc[1:] > 1e-12).any():
        raise ValueError("Ranking is inconsistent with RRF scores")
    if list(identity["rank"]) != list(range(1, 11)) or not identity.drug_name.equals(
            ranking.drug_name.head(10).reset_index(drop=True)):
        raise ValueError("Identity audit does not match the frozen Top-10")
    step("ranking_validated", candidates=len(ranking), top_n=10)

    candidates = []
    for row in ranking.head(10).itertuples():
        ledger_path = screen_dir / "evidence_ledger" / f"rank_{row.rank:02d}.json"
        ledger = _validated_ledger(ledger_path)
        if ledger.drug_name != row.drug_name or ledger.transcriptomic_rank != row.rank:
            raise ValueError(f"Ledger rank/name mismatch at rank {row.rank}")
        if ledger.trace_id != "luad-eh3226-landmark-v1":
            raise ValueError("Ledger source trace mismatch")
        if ledger.confidence_tier != "insufficient_evidence":
            raise ValueError("Unreviewed screen cannot upgrade evidence tier")
        audit = identity.iloc[row.rank - 1]
        candidates.append({
            "rank": int(row.rank), "drug_name": row.drug_name,
            "inferred_geo_pert_id": row.pert_id if pd.notna(row.pert_id) else None,
            "scores": {"spearman_reversal": float(row.spearman_reversal),
                       "connectivity": float(row.connectivity), "rrf": float(row.rrf)},
            "hub_identity_status": audit.hub_identity_status,
            "confidence_tier": ledger.confidence_tier,
            "supporting_citations": len(ledger.supporting_evidence),
            "contradicting_citations": len(ledger.contradicting_evidence),
            "context_citations": len(ledger.context_evidence),
            "ledger_path": str(ledger_path), "ledger_sha256": sha256_file(ledger_path),
        })
    triage_manifest_path = screen_dir / "literature_triage_manifest.json"
    if triage_manifest_path.exists():
        triage = _read_json(triage_manifest_path)
        _check_hash(screen_dir / "pubmed_search.json", triage["pubmed_search_sha256"])
        for entry in triage["reviewed_context"]:
            ledger = _validated_ledger(screen_dir / "evidence_ledger" /
                                       f"rank_{entry['rank']:02d}.json")
            if not any(c.identifier == entry["pmid"] for c in ledger.context_evidence):
                raise ValueError("Literature triage manifest and ledger differ")
    step("evidence_gate", accepted_as_efficacy=0, requiring_review=10,
         identity_status_counts=identity.hub_identity_status.value_counts().to_dict())
    route_question = choice_question("Which research task should be reviewed next?", {
        "resolve_identity": "Resolve mismatched or ambiguous compound IDs and stereochemistry.",
        "review_literature": "Review supporting and opposing LUAD efficacy and safety evidence.",
        "run_limma": "Check disease differential expression with paired limma.",
        "manual_review": "Inputs are too uncertain to choose a task automatically.",
    })
    model_usage = None
    model_name = None
    attempted_calls = 0
    if jev_client is None:
        route = gate_choice(route_question, None)
    else:
        attempted_calls = 1
        try:
            response = jev_client.ask({
                "disease": "LUAD", "paired_samples": 57,
                "top10_identity_status": identity.hub_identity_status.value_counts().to_dict(),
                "efficacy_evidence_curated": False, "limma_sensitivity_done": False,
            }, {"next_research_task": route_question})
            route = gate_choice(route_question, response["answers"]["next_research_task"])
            model_usage = response["usage"]
            model_name = response["model"]
        except Exception as exc:
            route = gate_choice(route_question, None)
            step("jev_unavailable", error_type=type(exc).__name__)
    step("next_task_routed", action=route.action, source=route.source,
         reason=route.reason, confidence=route.confidence)
    frozen = pd.read_csv("configs/luad_positive_controls.csv")
    if list(frozen.drug_name) != list(controls.drug_name):
        raise ValueError("Positive-control list differs from frozen preregistration")
    control_records = [{"drug_name": row.drug_name,
                        "rank": int(row.rank) if pd.notna(row.rank) else None}
                       for row in controls.itertuples()]
    step("controls_checked", prespecified=len(controls),
         measured=int(controls["rank"].notna().sum()))

    report = {
        "run_id": run_id, "status": "expression_screen_complete_evidence_pending",
        "disease": "LUAD", "data": {"paired_patients": 57, "a549_names": 4920,
                                  "landmark_genes": screen["landmark_genes_intersected"]},
        "method": screen["method"], "candidates": candidates,
        "disease_signature_sensitivity": sensitivity,
        "next_research_task": {"action": route.action, "source": route.source,
                               "reason": route.reason, "confidence": route.confidence,
                               "model": model_name},
        "prespecified_controls": control_records,
        "interpretation": "In vitro hypotheses only; no efficacy or treatment claim.",
        "limitations": ["EH3226 drug columns lack perturbagen IDs.",
                        "Drug-name annotations do not prove compound identity.",
                        "Paired t-test requires limma sensitivity analysis.",
                        "Literature, safety, and clinical context are not yet curated."],
        "cost": {"external_model_calls": attempted_calls,
                 "external_model_usd": 0 if attempted_calls == 0 else None,
                 "external_model_usage": model_usage,
                 "runtime_seconds": round(perf_counter() - started, 3)},
        "provenance": {"disease_manifest_sha256": sha256_file(disease_manifest),
                       "screen_manifest_sha256": sha256_file(screen_manifest),
                       "identity_audit_sha256": sha256_file(screen_dir / "identity_audit.csv")},
        "trace": trace,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "case_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    return report
