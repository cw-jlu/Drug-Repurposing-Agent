"""Audit seed-matched official-runner ablations of the frozen B2 fusion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from drug_repurposing_agent.data import sha256_file
from scripts.compare_recess_official_b2 import load_run, summarize


COMPONENTS = ("B0p", "B1k", "B1")


def paired(values: np.ndarray) -> dict:
    result = summarize(values)
    rng = np.random.default_rng(20260926)
    samples = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    return {**result,
            "mean_ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "b2_wins": int((values > 0).sum()), "ties": int((values == 0).sum())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", type=Path,
                        default=Path("artifacts/recess_official_components"))
    parser.add_argument("--b2", type=Path,
                        default=Path("benchmark/results/recess_official_b2"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/results/recess_official_component_ablation.json"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    reference = json.loads((repo / "benchmark/results/recess_official_b2_vs_11.json")
                           .read_text(encoding="utf-8"))
    seeds = reference["seed_order"]
    if len(seeds) != 100:
        raise ValueError("B2 reference must contain 100 ordered seeds")
    report = {
        "protocol": "RECeSS TRANSCRIPT official runner, N=100, K=5, ptest=0.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_code_commit": reference["official_code_commit"],
        "b2_implementation_sha256": reference["b2_implementation_sha256"],
        "component_implementation_sha256": sha256_file(
            repo / "benchmarks/recess_adapter/official_components.py"),
        "component_patch_sha256": sha256_file(
            repo / "benchmarks/recess_adapter/official_components.patch"),
        "seed_order": seeds,
        "selection_metric": "official five-fold rowwise AUC, best fold model",
        "splits": {},
    }
    for split in ("random_simple", "weakly_correlated"):
        runs = {"B2": load_run(args.b2.resolve(strict=True), "B2", split, seeds)}
        for component in COMPONENTS:
            runs[component] = load_run(
                args.components.resolve(strict=True) / f"results_{component}",
                component, split, seeds,
            )
        for method, run in runs.items():
            for metric, values in run["series"].items():
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite {metric} for {method}/{split}")
        report["splits"][split] = {
            "models": {method: {metric: summarize(values) for metric, values in run["series"].items()}
                       for method, run in runs.items()},
            "b2_minus_component": {
                component: {metric: paired(runs["B2"]["series"][metric] -
                                           runs[component]["series"][metric])
                            for metric in runs["B2"]["series"]}
                for component in COMPONENTS
            },
            "source_sha256": {method: run["source_files"] for method, run in runs.items()},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split, result in report["splits"].items():
        print(split)
        for method, metrics in result["models"].items():
            ns = metrics["NS-AUC"]
            print(f"{method:4} NS-AUC={ns['mean']:.4f} ± {ns['sd']:.4f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
