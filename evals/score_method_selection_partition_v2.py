"""Score frozen prescore choices against subsequently generated official outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.model_selector import METHODS
from drug_repurposing_agent.trace import TraceRecorder
from evals.grade_method_selection_traces import grade_choices


def load_outcomes(case: dict, runs: int, folds: int) -> tuple[dict[str, float], dict, dict[str, np.ndarray]]:
    case_root = Path(case["dataset_dir"]).parents[1]
    scores = {}
    hashes = {}
    vectors = {}
    seed_order = None
    for method in METHODS:
        folder = case_root / ("b2" if method == "B2" else "components") / f"results_{method}"
        suffix = f"{method}_TRANSCRIPT_random_simple_AUC_1.000000_{folds}_0.200000.csv"
        result_path = folder / f"results_N={runs}_{suffix}"
        seeds_path = folder / f"seeds_N={runs}_{suffix}"
        result = pd.read_csv(result_path, index_col=0)
        seeds = pd.read_csv(seeds_path, index_col=0)
        observed = [int(value) for value in seeds.loc["seed"].tolist()]
        if len(observed) != runs or len(set(observed)) != runs or result.shape[1] != runs:
            raise ValueError(f"Wrong run count for {case['id']}/{method}")
        if seed_order is None:
            seed_order = observed
        elif observed != seed_order:
            raise ValueError(f"Seeds differ for {case['id']}/{method}")
        values = pd.to_numeric(result.loc["Lin's AUC"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite NS-AUC for {case['id']}/{method}; no partial-case scoring")
        scores[method] = float(values.mean())
        vectors[method] = values
        hashes[method] = {"results": sha256_file(result_path), "seeds": sha256_file(seeds_path)}
    return scores, {"seed_order": seed_order, "source_sha256": hashes}, vectors


def score_choices(choices: list[dict], cases: list[dict], outcomes: dict[str, dict[str, float]],
                  repeats: int, seed_vectors: dict[str, dict[str, np.ndarray]] | None = None) -> dict:
    expected = {(case["id"], repeat) for case in cases for repeat in range(1, repeats + 1)}
    observed = [(row["case_id"], row["repeat"]) for row in choices]
    if len(observed) != len(expected) or set(observed) != expected or len(set(observed)) != len(observed):
        raise ValueError("Choices do not cover every case and repeat exactly once")
    rows = []
    for row in choices:
        scores = outcomes[row["case_id"]]
        selected = row["choice"]["method"]
        if selected not in METHODS or set(scores) != set(METHODS):
            raise ValueError("Unknown method in choices or outcomes")
        best = max(scores.values())
        rows.append({"case_id": row["case_id"], "repeat": row["repeat"],
                     "selected_method": selected, "selected_ns_auc": scores[selected],
                     "fixed_b2_ns_auc": scores["B2"], "oracle_ns_auc": best,
                     "selected_minus_fixed_b2": scores[selected] - scores["B2"],
                     "oracle_regret": best - scores[selected]})
    by_case = {}
    for case in cases:
        selected = [row for row in rows if row["case_id"] == case["id"]]
        by_case[case["id"]] = {
            "choices": [row["selected_method"] for row in selected],
            "mean_selected_ns_auc": mean(row["selected_ns_auc"] for row in selected),
            "fixed_b2_ns_auc": selected[0]["fixed_b2_ns_auc"],
            "oracle_ns_auc": selected[0]["oracle_ns_auc"],
            "mean_regret": mean(row["oracle_regret"] for row in selected),
        }
        if seed_vectors is not None:
            vectors = seed_vectors[case["id"]]
            if set(vectors) != set(METHODS):
                raise ValueError(f"Missing paired method vectors for {case['id']}")
            baseline = np.asarray(vectors["B2"], dtype=float)
            selected_vectors = [np.asarray(vectors[row["selected_method"]], dtype=float)
                                for row in selected]
            if (not baseline.size or any(vector.shape != baseline.shape for vector in selected_vectors)
                    or not np.isfinite(baseline).all()
                    or any(not np.isfinite(vector).all() for vector in selected_vectors)):
                raise ValueError(f"Nonfinite or unpaired seed vectors for {case['id']}")
            delta = np.mean(np.stack(selected_vectors), axis=0) - baseline
            by_case[case["id"]]["paired_seed_delta_vs_b2"] = {
                "mean": float(delta.mean()),
                "median": float(np.median(delta)),
                "seed_wins": int(np.sum(delta > 1e-12)),
                "seed_ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "seed_losses": int(np.sum(delta < -1e-12)),
                "seed_count": int(delta.size),
            }
    result = {"case_count": len(cases), "model_repeats_per_case": repeats,
            "mean_selected_ns_auc_across_cases": mean(x["mean_selected_ns_auc"] for x in by_case.values()),
            "mean_fixed_b2_ns_auc_across_cases": mean(x["fixed_b2_ns_auc"] for x in by_case.values()),
            "mean_oracle_regret_across_cases": mean(x["mean_regret"] for x in by_case.values()),
            "by_case": by_case, "trials": rows}
    if seed_vectors is not None:
        group_deltas = [row["paired_seed_delta_vs_b2"]["mean"] for row in by_case.values()]
        result["paired_partition_summary"] = {
            "mean_delta_vs_b2": float(mean(group_deltas)),
            "positive_partitions": sum(value > 1e-12 for value in group_deltas),
            "tied_partitions": sum(abs(value) <= 1e-12 for value in group_deltas),
            "negative_partitions": sum(value < -1e-12 for value in group_deltas),
            "interpretation": ("Descriptive across partitions of one parent dataset; "
                               "seed repeats and model repeats are not independent cohorts."),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        default=Path("configs/method_selection_partition_v2.json"))
    parser.add_argument("--cases", type=Path,
                        default=Path("configs/method_selection_partition_v2_cases.json"))
    parser.add_argument("--choices", type=Path,
                        default=Path("artifacts/reports/method_selection_partition_v2_choices.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/reports/method_selection_partition_v2_score.json"))
    args = parser.parse_args()
    trace = TraceRecorder("method_selection_scoring", args.output.parent / "traces")
    trace.emit("scoring_started", protocol=str(args.protocol), cases=str(args.cases),
               choices=str(args.choices))
    try:
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        choice_report = json.loads(args.choices.read_text(encoding="utf-8"))
        if (cases["protocol_sha256"] != sha256_file(args.protocol) or
                choice_report["protocol_sha256"] != sha256_file(args.protocol) or
                choice_report["cases_sha256"] != sha256_file(args.cases) or
                choice_report["status"] != "prescore_choices_frozen_outcomes_unseen"):
            raise ValueError("Frozen protocol, cases or prescore choices differ")
        choice_trace_grade = grade_choices(args.choices, args.cases, args.protocol)
        if choice_trace_grade["passed"] != choice_trace_grade["choices"]:
            raise ValueError("Provider-visible choice trace audit did not pass")
        trace.emit("choice_trace_verified", passed=choice_trace_grade["passed"],
                   choices=choice_trace_grade["choices"],
                   main_trace_sha256=choice_trace_grade["main_trace_sha256"])
        outcomes = {}
        receipts = {}
        seed_vectors = {}
        for case in cases["cases"]:
            outcomes[case["id"]], receipts[case["id"]], seed_vectors[case["id"]] = load_outcomes(
                case, protocol["outer_runs"], protocol["inner_folds"])
            trace.emit("case_outcome_verified", case_id=case["id"],
                       source_sha256=receipts[case["id"]]["source_sha256"])
        scores = score_choices(choice_report["choices"], cases["cases"], outcomes,
                               protocol["model_repeats_per_partition"], seed_vectors)
        report = {"status": "prospective_partition_evaluation_not_external_validation",
                  "scored_at": datetime.now(timezone.utc).isoformat(),
                  "protocol_sha256": sha256_file(args.protocol),
                  "cases_sha256": sha256_file(args.cases),
                  "prescore_choices_sha256": sha256_file(args.choices),
                  "prescore_choice_trace_grade": {
                      "passed": choice_trace_grade["passed"],
                      "choices": choice_trace_grade["choices"],
                      "main_trace_sha256": choice_trace_grade["main_trace_sha256"],
                  },
                  "outcomes": outcomes, "outcome_receipts": receipts,
                  "scores": scores, "trace_file": str(trace.path),
                  "limitation": protocol["disclosure"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"))
        trace.emit("scoring_completed", output=str(args.output),
                   output_sha256=sha256_file(args.output))
        print(f"Scored {scores['case_count']} partitions from frozen choices")
    except Exception as exc:
        trace.emit("scoring_failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
