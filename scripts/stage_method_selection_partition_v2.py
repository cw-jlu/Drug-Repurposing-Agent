"""Freeze label-blind disease partitions before model choice or outcome scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder


METHODS = ("B0p", "B1k", "B1", "B2")


def _frozen_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Frozen output differs: {path}")
        return
    path.write_bytes(content)


def stage(config_path: Path, source: Path, root: Path, cases_out: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("version") != "method_selection_partition_v2_prescore" or
            set(config.get("methods", {})) != set(METHODS) or
            config.get("partition_count") != 5 or config.get("split") != "random_simple"):
        raise ValueError("Frozen method-selection protocol changed")
    manifest_path = Path(config["source_manifest"])
    if sha256_file(manifest_path) != config["source_manifest_sha256"]:
        raise ValueError("Source manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("ratings_mat.csv", "items.csv", "users.csv"):
        if sha256_file(source / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Source data changed: {name}")
    ratings = pd.read_csv(source / "ratings_mat.csv", index_col=0)
    users = pd.read_csv(source / "users.csv", index_col=0)
    items = pd.read_csv(source / "items.csv", index_col=0, nrows=0)
    if list(ratings.columns) != list(users.columns) or list(ratings.index) != list(items.columns):
        raise ValueError("Source matrices are not aligned")
    groups: dict[int, list[str]] = {index: [] for index in range(5)}
    for disease_id in ratings.columns:
        digest = hashlib.sha256(f"{config['partition_salt']}:{disease_id}".encode()).digest()
        groups[int.from_bytes(digest[:8], "big") % 5].append(disease_id)
    if any(len(ids) < 15 for ids in groups.values()):
        raise ValueError("Partition unexpectedly small")
    cases = []
    for index, disease_ids in groups.items():
        case_id = f"transcript_disease_partition_{index + 1:02d}"
        dataset_dir = root / case_id / "datasets" / "TRANSCRIPT"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        item_target = dataset_dir / "items.csv"
        if not item_target.exists():
            try:
                os.link(source / "items.csv", item_target)
            except OSError:
                shutil.copy2(source / "items.csv", item_target)
        if sha256_file(item_target) != manifest["files"]["items.csv"]["sha256"]:
            raise ValueError("Staged item features differ")
        for name, frame in (("ratings_mat.csv", ratings[disease_ids]),
                            ("users.csv", users[disease_ids])):
            destination = dataset_dir / name
            csv = frame.to_csv(index_label=frame.index.name or "index").encode("utf-8")
            _frozen_write(destination, csv)
        blind_input = {
            "dataset": "TRANSCRIPT v2.0.0 disease partition",
            "partition_id": case_id,
            "drug_count": int(ratings.shape[0]),
            "disease_count": len(disease_ids),
            "expression_gene_count": int(users.shape[0]),
            "median_disease_signature_sd": round(float(users[disease_ids].std(axis=0).median()), 5),
            "split": config["split"],
            "split_description": "Hold out 20% of drug-disease pairs; the same drugs may occur in training",
            "methods": config["methods"],
            "metric_to_optimize": config["primary_metric"],
        }
        cases.append({"id": case_id, "blind_input": blind_input,
                      "disease_ids": disease_ids,
                      "dataset_dir": str(dataset_dir),
                      "staged_sha256": {name: sha256_file(dataset_dir / name)
                                        for name in ("ratings_mat.csv", "items.csv", "users.csv")}})
    result = {"version": config["version"], "protocol_sha256": sha256_file(config_path),
              "source_manifest_sha256": config["source_manifest_sha256"],
              "disclosure": config["disclosure"], "cases": cases}
    encoded = (json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    _frozen_write(cases_out, encoded)
    return {"case_file": str(cases_out), "case_file_sha256": sha256_file(cases_out),
            "case_count": len(cases), "partition_sizes": [len(case["disease_ids"]) for case in cases]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/method_selection_partition_v2.json"))
    parser.add_argument("--source", type=Path,
                        default=Path("data/raw/TRANSCRIPT_dataset_v2.0.0"))
    parser.add_argument("--root", type=Path,
                        default=Path("artifacts/method_selection_partition_v2"))
    parser.add_argument("--cases-out", type=Path,
                        default=Path("configs/method_selection_partition_v2_cases.json"))
    args = parser.parse_args()
    trace = TraceRecorder("method_selection_stage", args.root / "traces")
    trace.emit("staging_started", config=str(args.config), source=str(args.source))
    try:
        result = stage(args.config, args.source, args.root, args.cases_out)
        trace.emit("staging_completed", **result)
        print(json.dumps({**result, "trace_file": str(trace.path)}, ensure_ascii=False))
    except Exception as exc:
        trace.emit("staging_failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
