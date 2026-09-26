"""Trace-backed, idempotent preparation of the pinned RECeSS runner clone."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run
from scripts.run_recess_official_b2 import UPSTREAM_COMMIT


UPSTREAM_URL = "https://github.com/RECeSS-EU-Project/benchmark-code.git"
MARKERS = ('models.append("B2")', 'models.extend(["B0p", "B1k", "B1"])',
           "official_b2 import B2", "official_components")


def _git(upstream: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(upstream), *args], text=True).strip()


def prepare(upstream: Path, repo: Path, trace: TraceRecorder) -> dict:
    upstream = upstream.resolve()
    created = False
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        trace.emit("clone_started", url=UPSTREAM_URL, target=str(upstream))
        subprocess.run(["git", "clone", UPSTREAM_URL, str(upstream)], check=True)
        created = True
        trace.emit("clone_completed", target=str(upstream))
    elif not (upstream / ".git").is_dir():
        raise ValueError(f"Target exists and is not a Git clone: {upstream}")
    head = _git(upstream, "rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        if not created:
            raise ValueError(f"Existing clone is not at pinned commit {UPSTREAM_COMMIT}: {head}")
        subprocess.run(["git", "-C", str(upstream), "checkout", "--detach", UPSTREAM_COMMIT],
                       check=True)
        head = _git(upstream, "rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        raise ValueError("Pinned upstream checkout failed")
    trace.emit("upstream_verified", commit=head)

    source = upstream / "benchmark_pipeline.py"
    for marker, name in ((MARKERS[0], "official_b2.patch"),
                         (MARKERS[1], "official_components.patch")):
        current = source.read_text(encoding="utf-8")
        if marker in current:
            trace.emit("patch_already_present", patch=name)
            continue
        patch = repo / "benchmarks" / "recess_adapter" / name
        trace.emit("patch_check", patch=name, patch_sha256=sha256_file(patch))
        command = ["git", "-C", str(upstream), "apply", "--recount"]
        subprocess.run([*command, "--check", str(patch)], check=True)
        subprocess.run([*command, str(patch)], check=True)
        trace.emit("patch_applied", patch=name, patch_sha256=sha256_file(patch))
    current = source.read_text(encoding="utf-8")
    if any(marker not in current for marker in MARKERS):
        raise ValueError("Pinned runner is missing required B2/component hooks")
    changed = _git(upstream, "diff", "--name-only").splitlines()
    if set(changed) != {"benchmark_pipeline.py", "main.py"}:
        raise ValueError(f"Unexpected changes in pinned upstream: {changed}")
    subprocess.run(["git", "-C", str(upstream), "diff", "--check"], check=True)
    result = {"upstream": str(upstream), "commit": head,
              "patched_files": changed,
              "benchmark_pipeline_sha256": sha256_file(source),
              "main_sha256": sha256_file(upstream / "main.py")}
    trace.emit("upstream_prepared", **result)
    return result


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path,
                        default=Path("artifacts/upstream_recess_benchmark_code"))
    args = parser.parse_args()
    result = prepare(args.upstream, Path(__file__).resolve().parents[1], trace)
    print(f"Pinned RECeSS runner ready at {result['upstream']}")


def main() -> None:
    traced_run("recess_official_upstream_prepare", _main)


if __name__ == "__main__":
    main()
