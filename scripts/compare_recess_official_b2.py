"""Pair official B2 runs with all 11 published TRANSCRIPT reference models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from drug_repurposing_agent.data import sha256_file


PUBLISHED_COMMIT = "cf5d9fdcb1ccd1676c7a0e7a39e784d79557fae0"
MODELS = ("ALSWR", "BNNR", "DDA_SKF", "FastaiCollabWrapper", "HAN",
          "LibMF", "LogisticMF", "MBiRW", "NIMCGCN", "PMF", "SCPMF")
METRICS = {"NS-AUC": "Lin's AUC", "global AUC": "global AUC",
           "global NDCG": "global NDCG"}


def result_paths(folder: Path, model: str, split: str) -> tuple[Path, Path]:
    name = f"{model}_TRANSCRIPT_{split}_AUC_1.000000_5_0.200000.csv"
    result = folder / f"results_N=100_{name}"
    seeds = folder / f"seeds_N=100_{name}"
    if not result.is_file() or not seeds.is_file():
        raise FileNotFoundError(f"100-run, 5-fold files missing for {model}/{split}: {folder}")
    return result, seeds


def load_run(folder: Path, model: str, split: str, expected_seeds: list[int]) -> dict:
    result_path, seed_path = result_paths(folder, model, split)
    result = pd.read_csv(result_path, index_col=0)
    seeds = pd.read_csv(seed_path, index_col=0)
    observed_seeds = [int(value) for value in seeds.loc["seed"].tolist()]
    if observed_seeds != expected_seeds or result.shape[1] != 100:
        raise ValueError(f"Seed order or run count differs for {model}/{split}")
    series = {}
    for label, source_row in METRICS.items():
        if source_row not in result.index:
            raise ValueError(f"Metric {source_row} missing for {model}/{split}")
        values = pd.to_numeric(result.loc[source_row], errors="coerce").to_numpy(dtype=float)
        series[label] = values
    return {
        "series": series,
        "source_files": {result_path.name: sha256_file(result_path),
                         seed_path.name: sha256_file(seed_path)},
    }


def summarize(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"n": 0, "mean": None, "sd": None, "median": None}
    return {"n": int(len(finite)), "mean": float(finite.mean()),
            "sd": float(finite.std(ddof=1)) if len(finite) > 1 else None,
            "median": float(np.median(finite))}


def paired_summary(difference: np.ndarray) -> dict:
    values = difference[np.isfinite(difference)]
    result = summarize(values)
    if not len(values):
        result.update({"mean_ci95": None, "b2_wins": 0, "ties": 0})
        return result
    rng = np.random.default_rng(20260925)
    samples = values[rng.integers(0, len(values), size=(5000, len(values)))].mean(axis=1)
    result.update({"mean_ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
                   "b2_wins": int((values > 0).sum()),
                   "ties": int((values == 0).sum())})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published", type=Path, required=True,
                        help="Clone of RECeSS-EU-Project/benchmark-results")
    parser.add_argument("--ours", type=Path,
                        default=Path("artifacts/recess_official_b2/results_B2"))
    parser.add_argument("--data", type=Path,
                        default=Path("data/raw/TRANSCRIPT_dataset_v2.0.0"))
    parser.add_argument("--output", type=Path,
                        default=Path("benchmark/results/recess_official_b2_vs_11.json"))
    args = parser.parse_args()
    published = args.published.resolve(strict=True)
    published_head = subprocess.check_output(
        ["git", "-C", str(published), "rev-parse", "HEAD"], text=True
    ).strip()
    if published_head != PUBLISHED_COMMIT:
        raise ValueError(f"Expected published results commit {PUBLISHED_COMMIT}; got {published_head}")
    expected_seeds = np.random.RandomState(1234).choice(range(int(1e8)), size=100).tolist()
    repo = Path(__file__).resolve().parents[1]
    input_data = args.data.resolve(strict=True)
    input_hashes = {
        name: sha256_file(input_data / name)
        for name in ("ratings_mat.csv", "items.csv", "users.csv")
    }
    manifest = json.loads((repo / "data" / "manifests" /
                           "transcript-v2.0.0.json").read_text(encoding="utf-8"))
    for name, digest in input_hashes.items():
        if digest != manifest["files"][name]["sha256"]:
            raise ValueError(f"TRANSCRIPT input differs from the pinned manifest: {name}")
    report = {
        "protocol": "RECeSS TRANSCRIPT official runner, N=100, K=5, ptest=0.2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "local_runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "stanscofi": version("stanscofi"),
            "benchscofi": version("benchscofi"),
            "cute-ranking": version("cute-ranking"),
        },
        "official_code_commit": "a7f11077271cedf3a98a82e3dc74b6fc0e93986e",
        "published_results_commit": PUBLISHED_COMMIT,
        "selection_metric": "official five-fold rowwise AUC, best fold model",
        "primary_metric": "Lin's AUC (paper NS-AUC)",
        "b2_parameters": {"neighbors": 10, "rrf_k": 60},
        "b2_implementation_sha256": sha256_file(
            repo / "benchmarks" / "recess_adapter" / "official_b2.py"),
        "runner_patch_sha256": sha256_file(
            repo / "benchmarks" / "recess_adapter" / "official_b2.patch"),
        "transcript_input_sha256": input_hashes,
        "seed_order": expected_seeds,
        "splits": {},
    }
    for split in ("random_simple", "weakly_correlated"):
        reference_dir = published / ("results_TRANSCRIPT" if split == "random_simple"
                                     else "results_TRANSCRIPT_weakly_correlated")
        b2 = load_run(args.ours.resolve(strict=True), "B2", split, expected_seeds)
        rows = {"B2": {metric: summarize(values) for metric, values in b2["series"].items()}}
        hashes = {"B2": b2["source_files"]}
        paired = {}
        for model in MODELS:
            reference = load_run(reference_dir / f"results_{model}", model,
                                 split, expected_seeds)
            rows[model] = {metric: summarize(values)
                           for metric, values in reference["series"].items()}
            hashes[model] = reference["source_files"]
            deltas = {}
            for metric in METRICS:
                ours = b2["series"][metric]
                theirs = reference["series"][metric]
                overlap = np.isfinite(ours) & np.isfinite(theirs)
                deltas[metric] = paired_summary(ours[overlap] - theirs[overlap])
            paired[model] = deltas
        report["splits"][split] = {"models": rows, "b2_minus_reference": paired,
                                   "source_sha256": hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for split, contents in report["splits"].items():
        print(split)
        for model, metrics in sorted(contents["models"].items(),
                                     key=lambda item: (-(item[1]["NS-AUC"]["mean"]
                                                         if item[1]["NS-AUC"]["mean"] is not None
                                                         else -1), item[0])):
            value = metrics["NS-AUC"]
            display = f"{value['mean']:.4f}" if value["mean"] is not None else "NA"
            print(f"{model:24} NS-AUC={display} n={value['n']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
