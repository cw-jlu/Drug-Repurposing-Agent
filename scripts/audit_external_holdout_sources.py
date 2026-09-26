"""Read-only provenance/availability gate for candidate external v3 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


TREE_URL = "https://api.github.com/repos/enockniyonkuru/cdrpipe-comparative-analysis/git/trees/main?recursive=1"
OUTPUT = Path("artifacts/reports/external_holdout_source_audit.json")


def _main(trace: TraceRecorder) -> None:
    request = Request(TREE_URL, headers={"User-Agent": "drug-agent-data-audit",
                                         "Accept": "application/vnd.github+json"})
    trace.emit("source_inventory_requested", url=TREE_URL)
    with urlopen(request, timeout=30) as response:
        tree = json.loads(response.read().decode("utf-8"))
    if tree.get("truncated"):
        raise RuntimeError("GitHub tree response was truncated")
    blobs = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
    disease_files = [name for name in blobs
                     if name.startswith("creeds/data/manual_signatures_extracted/")
                     and name.endswith(".csv")]
    required = ["drug_signatures/data/cmap/cmap_signatures.RData",
                "drug_evidence/data/open_targets/known_drug_info_data.parquet"]
    matrix_or_label_missing = [name for name in required if name not in blobs]
    report = {
        "status": "external_holdout_not_ready_no_model_outcomes_computed",
        "audited_source": "CDRPipe comparative analysis / CREEDS + CMap + Open Targets",
        "repository": "https://github.com/enockniyonkuru/cdrpipe-comparative-analysis",
        "github_tree_sha": tree["sha"], "github_tree_url": TREE_URL,
        "disease_signature_csv_count": len(disease_files),
        "tracked_drug_metadata_present": "drug_signatures/data/cmap/cmap_drug_experiments_new.csv" in blobs,
        "required_large_files_absent_from_git": matrix_or_label_missing,
        "decision": "Do not claim or run a v3 official independent holdout until raw drug-expression vectors, disease-expression vectors, indication labels, ID mapping, and frozen split/label policy are staged and hash-verified.",
        "alternative_source": {
            "name": "TRACE cross-disease dataset",
            "url": "https://data.mendeley.com/datasets/fy93xjjrb8/1",
            "published_disease_count": 3,
            "gate": "insufficient independent disease-level units for the planned multi-unit selector comparison; published drug_scores are outcomes, not raw perturbation expression features"
        },
        "scope": "Metadata availability audit only; no dataset download or performance evaluation"
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("source_audit_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               github_tree_sha=tree["sha"], disease_signature_csv_count=len(disease_files),
               absent=matrix_or_label_missing, ready=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    traced_run("external_holdout_source_audit", _main)
