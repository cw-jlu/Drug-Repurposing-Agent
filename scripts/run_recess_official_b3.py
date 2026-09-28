"""Run the frozen B3 row-fusion models in the pinned official RECeSS runner.

Apply official_b2.patch, official_components.patch and official_b3.patch to the
pinned upstream clone first. Configuration: configs/b3_row_fusion_v1.json.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_recess_official_b2 import UPSTREAM_COMMIT, prepare_dataset
from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


COMPONENTS = ("B3", "B3noREV")


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data", type=Path,
                        default=Path("data/raw/TRANSCRIPT_dataset_v2.0.0"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/recess_official_b3"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--njobs", type=int, default=1)
    parser.add_argument("--models", default=",".join(COMPONENTS))
    parser.add_argument("--splitting", choices=["random_simple", "weakly_correlated",
                                               "random_simple,weakly_correlated"],
                        default="random_simple,weakly_correlated")
    args = parser.parse_args()
    trace.emit("parameters", upstream=str(args.upstream), data=str(args.data),
               output=str(args.output), n=args.n, k=args.k, njobs=args.njobs,
               models=args.models, splitting=args.splitting)
    models = args.models.split(",")
    if not models or len(set(models)) != len(models) or any(model not in COMPONENTS for model in models):
        parser.error("--models must be a nonempty, unique subset of B3,B3noREV")
    if args.n < 1 or args.k < 2 or args.njobs < 1:
        parser.error("N and njobs must be positive and K at least two")
    upstream = args.upstream.resolve(strict=True)
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream commit {UPSTREAM_COMMIT}; got {head}")
    trace.emit("upstream_verified", commit=head)
    source = (upstream / "benchmark_pipeline.py").read_text(encoding="utf-8")
    if 'models.extend(["B3", "B3noREV"])' not in source or "official_b3" not in source:
        raise ValueError("Apply official_b2, official_components and official_b3 patches to upstream first")
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    prepare_dataset(args.data.resolve(strict=True), output / "datasets" / "TRANSCRIPT")
    trace.emit("dataset_staged", files={name: sha256_file(output / "datasets" / "TRANSCRIPT" / name)
                                        for name in ("ratings_mat.csv", "items.csv", "users.csv")})
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(repo), str(repo / "src"), environment.get("PYTHONPATH", "")) if value
    )
    environment["MPLBACKEND"] = "Agg"
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    command = [sys.executable, "-m", "main", "--models", ",".join(models),
               "--datasets", "TRANSCRIPT", "--splitting", args.splitting,
               "--N", str(args.n), "--K", str(args.k), "--njobs", str(args.njobs),
               "--save_folder", str(output)]
    trace.emit("official_runner_started", command=command, cwd=str(upstream))
    subprocess.run(command, cwd=upstream, env=environment, check=True)
    trace.emit("official_runner_finished")
    for model in models:
        for split in args.splitting.split(","):
            folder = output / f"results_{model}"
            suffix = f"_{model}_TRANSCRIPT_{split}_AUC_1.000000_{args.k}_0.200000.csv"
            for prefix in ("results", "seeds"):
                result = folder / f"{prefix}_N={args.n}{suffix}"
                if not result.is_file():
                    raise FileNotFoundError(result)
                trace.emit("result_verified", model=model, split=split, kind=prefix,
                           path=str(result), sha256=sha256_file(result))
                print(f"Official {model} {prefix}: {result}")


def main() -> None:
    traced_run("recess_official_b3", _main)


if __name__ == "__main__":
    main()
