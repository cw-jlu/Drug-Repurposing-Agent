"""Gate official partition outcomes on frozen, trace-verified LLM choices.

The default mode is a read-only preflight. --execute runs the pinned official
RECeSS wrappers only after all fifteen provider-visible choices pass audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run
from evals.grade_method_selection_traces import grade_choices
from scripts.run_recess_official_b2 import UPSTREAM_COMMIT


DATA_FILES = ("ratings_mat.csv", "items.csv", "users.csv")
METHODS = ("B0p", "B1k", "B1", "B2")


def runner_commands(repo: Path, upstream: Path, case: dict, runs: int,
                    folds: int) -> list[list[str]]:
    """Build the two official-runner commands without touching outcomes."""
    dataset = (repo / case["dataset_dir"]).resolve()
    case_root = dataset.parents[1]
    common = ["--upstream", str(upstream), "--data", str(dataset),
              "--n", str(runs), "--k", str(folds), "--splitting", "random_simple"]
    return [
        [sys.executable, "-m", "scripts.run_recess_official_b2", *common,
         "--output", str(case_root / "b2")],
        [sys.executable, "-m", "scripts.run_recess_official_components", *common,
         "--output", str(case_root / "components"), "--models", "B0p,B1k,B1"],
    ]


def preflight(repo: Path, protocol_path: Path, cases_path: Path,
              upstream: Path) -> dict:
    """Verify frozen data and runner state without reading association outcomes."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if (cases.get("protocol_sha256") != sha256_file(protocol_path) or
            protocol.get("partition_count") != len(cases.get("cases", [])) or
            tuple(protocol.get("methods", {})) != METHODS or
            protocol.get("split") != "random_simple" or
            protocol.get("outer_runs") != 20 or protocol.get("inner_folds") != 5):
        raise ValueError("Frozen partition protocol or case inventory differs")
    if not upstream.is_dir():
        raise FileNotFoundError(f"Pinned upstream clone missing: {upstream}")
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                                   text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError(f"Pinned upstream commit differs: {head}")
    pipeline = (upstream / "benchmark_pipeline.py").read_text(encoding="utf-8")
    if ('models.append("B2")' not in pipeline or
            'models.extend(["B0p", "B1k", "B1"])' not in pipeline or
            "official_components" not in pipeline or "official_b2 import B2" not in pipeline):
        raise ValueError("Pinned upstream is missing the two official patches")
    subprocess.run(["git", "-C", str(upstream), "diff", "--check"], check=True)

    staged_root = (repo / "artifacts" / "method_selection_partition_v2").resolve()
    seen_diseases: set[str] = set()
    planned = []
    any_outcome_dir = False
    for case in cases["cases"]:
        case_id = case["id"]
        dataset = (repo / case["dataset_dir"]).resolve()
        expected = staged_root / case_id / "datasets" / "TRANSCRIPT"
        if dataset != expected or not dataset.is_dir():
            raise ValueError(f"Staged dataset path differs for {case_id}")
        if (len(case["disease_ids"]) != case["blind_input"]["disease_count"] or
                seen_diseases.intersection(case["disease_ids"])):
            raise ValueError(f"Overlapping or mismatched disease group: {case_id}")
        seen_diseases.update(case["disease_ids"])
        for name in DATA_FILES:
            if sha256_file(dataset / name) != case["staged_sha256"][name]:
                raise ValueError(f"Staged input hash differs: {case_id}/{name}")
        case_root = dataset.parents[1]
        outcome_dirs = {name: (case_root / name).exists() for name in ("b2", "components")}
        any_outcome_dir |= any(outcome_dirs.values())
        planned.append({"case_id": case_id, "dataset_dir": str(dataset),
                        "outcome_dirs_exist": outcome_dirs,
                        "commands": runner_commands(repo, upstream, case,
                                                    protocol["outer_runs"],
                                                    protocol["inner_folds"])})
    if len(seen_diseases) != 151:
        raise ValueError("Frozen disease partitions do not cover 151 unique IDs")
    return {"protocol_sha256": sha256_file(protocol_path),
            "cases_sha256": sha256_file(cases_path),
            "upstream_commit": head,
            "upstream_pipeline_sha256": sha256_file(upstream / "benchmark_pipeline.py"),
            "partition_count": len(planned), "unique_diseases": len(seen_diseases),
            "outer_runs": protocol["outer_runs"], "inner_folds": protocol["inner_folds"],
            "method_seed_runs": len(planned) * len(METHODS) * protocol["outer_runs"],
            "any_outcome_dir": any_outcome_dir, "plan": planned}


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        default=Path("configs/method_selection_partition_v2.json"))
    parser.add_argument("--cases", type=Path,
                        default=Path("configs/method_selection_partition_v2_cases.json"))
    parser.add_argument("--choices", type=Path,
                        default=Path("artifacts/reports/method_selection_partition_v2_choices.json"))
    parser.add_argument("--upstream", type=Path,
                        default=Path("artifacts/upstream_recess_benchmark_code"))
    parser.add_argument("--score-output", type=Path,
                        default=Path("artifacts/reports/method_selection_partition_v2_score.json"))
    parser.add_argument("--execute", action="store_true",
                        help="Run 5 partitions × 4 methods × 20 seeds after choice trace audit")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    upstream = args.upstream.resolve()
    trace.emit("preflight_started", execute=args.execute, protocol=str(args.protocol),
               cases=str(args.cases), upstream=str(upstream))
    summary = preflight(repo, args.protocol, args.cases, upstream)
    trace.emit("preflight_passed", **{key: value for key, value in summary.items()
                                      if key != "plan"}, choices_present=args.choices.is_file())
    if not args.execute:
        print(json.dumps({"status": "preflight_only_no_outcomes_read_or_computed",
                          "partitions": summary["partition_count"],
                          "method_seed_runs_if_executed": summary["method_seed_runs"],
                          "choices_present": args.choices.is_file(),
                          "outcome_dirs_present": summary["any_outcome_dir"],
                          "upstream_commit": summary["upstream_commit"]}, ensure_ascii=False))
        return
    if summary["any_outcome_dir"]:
        raise ValueError("Partition outcome directory already exists; refusing overwrite or partial rerun")
    if args.score_output.exists():
        raise ValueError("Partition score output already exists; refusing overwrite")
    if not args.choices.is_file():
        raise FileNotFoundError("Freeze all fifteen LLM choices before official outcomes")
    grade = grade_choices(args.choices, args.cases, args.protocol)
    if grade["choices"] != 15 or grade["passed"] != 15:
        raise ValueError("All fifteen provider-visible method choices must pass trace grading")
    trace.emit("choices_locked", choice_report_sha256=sha256_file(args.choices),
               main_trace_sha256=grade["main_trace_sha256"],
               provider_trace_hashes=[row["provider_trace_sha256"] for row in grade["rows"]])
    for case in summary["plan"]:
        for command in case["commands"]:
            trace.emit("official_runner_started", case_id=case["case_id"], command=command)
            subprocess.run(command, cwd=repo, check=True)
            trace.emit("official_runner_finished", case_id=case["case_id"],
                       wrapper=command[2])
    score_command = [sys.executable, "-m", "evals.score_method_selection_partition_v2",
                     "--protocol", str(args.protocol), "--cases", str(args.cases),
                     "--choices", str(args.choices), "--output", str(args.score_output)]
    trace.emit("scoring_started", command=score_command)
    subprocess.run(score_command, cwd=repo, check=True)
    trace.emit("scoring_finished", output=str(args.score_output),
               output_sha256=sha256_file(args.score_output))
    print(f"Official partition comparison saved to {args.score_output}")


def main() -> None:
    traced_run("method_selection_partition_v2_benchmarks", _main)


if __name__ == "__main__":
    main()
