"""Run B2 inside the pinned, minimally patched upstream RECeSS runner.

Apply benchmarks/recess_adapter/official_b2.patch to benchmark-code commit
a7f11077271cedf3a98a82e3dc74b6fc0e93986e before invoking this script.
The default N=100, K=5 and dataset seed=1234 follow upstream main.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from drug_repurposing_agent.data import sha256_file


UPSTREAM_COMMIT = "a7f11077271cedf3a98a82e3dc74b6fc0e93986e"
DATA_FILES = ("ratings_mat.csv", "items.csv", "users.csv")


def prepare_dataset(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        original = source / name
        staged = destination / name
        if not original.is_file():
            raise FileNotFoundError(original)
        if staged.exists():
            if sha256_file(staged) != sha256_file(original):
                raise ValueError(f"Staged TRANSCRIPT input differs from source: {staged}")
            continue
        try:
            os.link(original, staged)
        except OSError:
            shutil.copy2(original, staged)
        if sha256_file(staged) != sha256_file(original):
            raise ValueError(f"Staged TRANSCRIPT input failed verification: {staged}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True,
                        help="Clone of RECeSS-EU-Project/benchmark-code at the pinned commit")
    parser.add_argument("--data", type=Path,
                        default=Path("data/raw/TRANSCRIPT_dataset_v2.0.0"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/recess_official_b2"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--splitting", choices=["random_simple", "weakly_correlated",
                                               "random_simple,weakly_correlated"],
                        default="random_simple,weakly_correlated")
    args = parser.parse_args()
    if args.n < 1 or args.k < 2:
        parser.error("N must be positive and K must be at least two")
    upstream = args.upstream.resolve(strict=True)
    upstream_head = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream_head != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream commit {UPSTREAM_COMMIT}; got {upstream_head}")
    pipeline_source = (upstream / "benchmark_pipeline.py").read_text(encoding="utf-8")
    if 'models.append("B2")' not in pipeline_source or "official_b2 import B2" not in pipeline_source:
        raise ValueError("Apply official_b2.patch to the upstream clone first")
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    prepare_dataset(args.data.resolve(strict=True), output / "datasets" / "TRANSCRIPT")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(repo), str(repo / "src"), environment.get("PYTHONPATH", ""))
        if value
    )
    environment["MPLBACKEND"] = "Agg"
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    command = [sys.executable, "-m", "main", "--models", "B2", "--datasets",
               "TRANSCRIPT", "--splitting", args.splitting, "--N", str(args.n),
               "--K", str(args.k), "--njobs", "1", "--save_folder", str(output)]
    subprocess.run(command, cwd=upstream, env=environment, check=True)
    for split in args.splitting.split(","):
        result = (output / "results_B2" /
                  f"results_N={args.n}_B2_TRANSCRIPT_{split}_AUC_1.000000_{args.k}_0.200000.csv")
        seed_file = result.with_name(result.name.replace("results_N=", "seeds_N="))
        if not result.is_file() or not seed_file.is_file():
            raise FileNotFoundError(f"Official runner output missing: {result} or {seed_file}")
        print(f"Official B2 result: {result}")
        print(f"Official B2 seeds: {seed_file}")


if __name__ == "__main__":
    main()
