import json
from pathlib import Path

import pytest

from scripts.run_method_selection_partition_v2_benchmarks import preflight, runner_commands


def test_runner_commands_keep_the_same_dataset_and_official_parameters(tmp_path: Path):
    repo = tmp_path / "repo"
    upstream = tmp_path / "upstream"
    case = {"dataset_dir": "artifacts/partitions/p1/datasets/TRANSCRIPT"}
    commands = runner_commands(repo, upstream, case, 20, 5)
    assert len(commands) == 2
    assert commands[0][2] == "scripts.run_recess_official_b2"
    assert commands[1][2] == "scripts.run_recess_official_components"
    for command in commands:
        assert command[command.index("--n") + 1] == "20"
        assert command[command.index("--k") + 1] == "5"
        assert command[command.index("--splitting") + 1] == "random_simple"
        assert command[command.index("--data") + 1] == str(
            (repo / case["dataset_dir"]).resolve())
    assert commands[1][-2:] == ["--models", "B0p,B1k,B1"]


def test_preflight_rejects_changed_protocol_before_runner_access(tmp_path: Path):
    protocol = tmp_path / "protocol.json"
    cases = tmp_path / "cases.json"
    protocol.write_text(json.dumps({"partition_count": 5, "methods": {
        "B0p": {}, "B1k": {}, "B1": {}, "B2": {}},
        "split": "random_simple", "outer_runs": 19, "inner_folds": 5}), encoding="utf-8")
    cases.write_text(json.dumps({"protocol_sha256": "wrong", "cases": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen partition protocol"):
        preflight(tmp_path, protocol, cases, tmp_path / "missing_upstream")
