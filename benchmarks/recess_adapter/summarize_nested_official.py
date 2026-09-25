"""Aggregate the saved official nested-CV runs and parameter frequencies."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/results/nested_cv_official_summary.json"))
    args = parser.parse_args()
    paths = sorted(args.results.glob("nested_cv_official_*_seed*.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(runs) != 10:
        raise ValueError(f"Expected 10 runs, found {len(runs)}")
    summary = {
        "protocol": "official_matrix_factorization_nested_cv_v1",
        "summarized_at": datetime.now(timezone.utc).isoformat(),
        "outer_seeds": sorted({run["seed"] for run in runs}),
        "result_files": [str(path) for path in paths],
        "splits": {},
        "leakage_audit_passed": all(
            run["leakage_audit"]["outer_train_test_overlap"] == 0
            and run["leakage_audit"]["outer_test_coordinates_seen_by_inner_cv"] == 0
            and run["leakage_audit"]["prediction_dataset_labels"] == "all_zero"
            for run in runs
        ),
    }
    for split in ("random_simple", "weakly_correlated"):
        selected_runs = [run for run in runs if run["split"] == split]
        methods = {}
        for method in ("ALSWR", "PMF", "LogisticMF"):
            auc = np.array([
                run["methods"][method]["outer_test"]["global_AUC"]
                for run in selected_runs
            ])
            ndcg = np.array([
                run["methods"][method]["outer_test"]["global_NDCG"]
                for run in selected_runs
            ])
            choices = Counter(
                json.dumps(run["methods"][method]["selected_parameters"], sort_keys=True)
                for run in selected_runs
            )
            methods[method] = {
                "runs": len(selected_runs),
                "global_AUC_mean": float(auc.mean()),
                "global_AUC_sd": float(auc.std(ddof=1)),
                "global_NDCG_mean": float(ndcg.mean()),
                "global_NDCG_sd": float(ndcg.std(ddof=1)),
                "selected_parameter_frequency": [
                    {"parameters": json.loads(key), "count": count}
                    for key, count in choices.most_common()
                ],
            }
        summary["splits"][split] = {"runs": len(selected_runs), "methods": methods}
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["splits"], indent=2))


if __name__ == "__main__":
    main()
