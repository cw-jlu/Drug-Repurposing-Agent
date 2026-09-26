"""Paired, retrospective audit of the two frozen split-level method choices.

This does not turn two choices into an independent selector evaluation. In
particular, weakly_correlated reuses one outer test set across all 100 seeds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.model_selector import METHODS, SPLITS
from drug_repurposing_agent.trace import TraceRecorder


def result_files(root: Path, method: str, split: str) -> tuple[Path, Path]:
    base = root / ("recess_official_b2" if method == "B2" else "recess_official_components")
    if method != "B2":
        base /= f"results_{method}"
    suffix = f"{method}_TRANSCRIPT_{split}_AUC_1.000000_5_0.200000.csv"
    return (base / f"results_N=100_{suffix}", base / f"seeds_N=100_{suffix}")


def load_seed_vector(result: Path, seed_file: Path) -> tuple[list[int], np.ndarray]:
    scores = pd.read_csv(result, index_col=0)
    seeds = pd.read_csv(seed_file, index_col=0)
    if "Lin's AUC" not in scores.index or "seed" not in seeds.index:
        raise ValueError(f"Missing official NS-AUC or seed row: {result}")
    values = pd.to_numeric(scores.loc["Lin's AUC"], errors="raise").to_numpy(dtype=float)
    raw_seeds = pd.to_numeric(seeds.loc["seed"], errors="raise").to_numpy(dtype=float)
    if (len(values) != 100 or len(raw_seeds) != 100 or
            not np.isfinite(values).all() or not np.isfinite(raw_seeds).all() or
            not np.equal(raw_seeds, np.floor(raw_seeds)).all() or
            len(set(raw_seeds)) != 100):
        raise ValueError(f"Expected 100 finite unique-seed official results: {result}")
    return [int(seed) for seed in raw_seeds], values


def paired_summary(selected: np.ndarray, baseline: np.ndarray) -> dict:
    if selected.shape != baseline.shape or not np.isfinite(selected).all() or not np.isfinite(baseline).all():
        raise ValueError("Paired score vectors differ or contain nonfinite values")
    differences = selected - baseline
    return {
        "selected_mean": float(np.mean(selected)),
        "fixed_b2_mean": float(np.mean(baseline)),
        "paired_mean_delta": float(np.mean(differences)),
        "paired_median_delta": float(np.median(differences)),
        "paired_delta_p05": float(np.quantile(differences, 0.05)),
        "paired_delta_p95": float(np.quantile(differences, 0.95)),
        "seed_wins": int(np.sum(differences > 1e-12)),
        "seed_ties": int(np.sum(np.abs(differences) <= 1e-12)),
        "seed_losses": int(np.sum(differences < -1e-12)),
        "seed_count": int(len(differences)),
    }


def audit(choices_path: Path, input_path: Path, root: Path) -> dict:
    choices = json.loads(choices_path.read_text(encoding="utf-8"))
    if choices.get("input_sha256") != sha256_file(input_path):
        raise ValueError("Frozen selector input hash differs")
    selected = choices.get("choices")
    if not isinstance(selected, dict) or set(selected) != set(SPLITS):
        raise ValueError("Frozen selector choices are incomplete")
    splits = {}
    for split in SPLITS:
        method = selected[split]["method"]
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        vectors = {}
        receipt = {}
        seed_order = None
        for candidate in METHODS:
            result, seeds = result_files(root, candidate, split)
            observed_seeds, vector = load_seed_vector(result, seeds)
            if seed_order is None:
                seed_order = observed_seeds
            elif observed_seeds != seed_order:
                raise ValueError(f"Unpaired official seeds for {split}/{candidate}")
            vectors[candidate] = vector
            receipt[candidate] = {"result_sha256": sha256_file(result),
                                  "seeds_sha256": sha256_file(seeds)}
        assert seed_order is not None
        method_means = {name: float(np.mean(values)) for name, values in vectors.items()}
        oracle = max(method_means, key=method_means.get)
        splits[split] = {
            "selected_method": method,
            "method_ns_auc_means": method_means,
            "best_fixed_method_in_hindsight": oracle,
            "selected_is_best_fixed_method": method == oracle,
            "selected_vs_b2": paired_summary(vectors[method], vectors["B2"]),
            "seed_order": seed_order,
            "file_receipts": receipt,
            "test_dependence": ("Repeated outer runs share one held-out test set; seed deltas "
                                "are not independent external validations" if split == "weakly_correlated"
                                else "Repeated splits of the same parent dataset; seed deltas are descriptive"),
        }
    return {"status": "retrospective_two_choice_audit_not_generalization",
            "audited_at": datetime.now(timezone.utc).isoformat(),
            "selector_report_sha256": sha256_file(choices_path),
            "selector_input_sha256": sha256_file(input_path),
            "splits": splits,
            "limitation": ("Only two prior LLM choices across one dataset; selected methods were "
                           "compared after their official outputs existed. No efficacy or "
                           "generalizable selector improvement is established.")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--choices", type=Path,
                        default=Path("benchmark/results/recess_component_llm_selector_v1.json"))
    parser.add_argument("--input", type=Path, default=Path("configs/component_selector_v1.json"))
    parser.add_argument("--results-root", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/reports/historical_selector_v1_paired.json"))
    args = parser.parse_args()
    trace = TraceRecorder("historical_selector_paired_audit", args.output.parent / "traces")
    trace.emit("audit_started", choices=str(args.choices), input=str(args.input),
               results_root=str(args.results_root))
    try:
        report = audit(args.choices, args.input, args.results_root)
        report["trace_file"] = str(trace.path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"))
        trace.emit("audit_completed", output=str(args.output),
                   output_sha256=sha256_file(args.output),
                   choices={split: row["selected_method"] for split, row in report["splits"].items()})
        for split, row in report["splits"].items():
            delta = row["selected_vs_b2"]["paired_mean_delta"]
            print(f"{split}: {row['selected_method']} vs B2 = {delta:+.4f} NS-AUC (descriptive)")
    except Exception as exc:
        trace.emit("audit_failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    main()
