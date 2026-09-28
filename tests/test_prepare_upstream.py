from pathlib import Path

import pytest

from drug_repurposing_agent.trace import TraceRecorder
from scripts.prepare_recess_official_upstream import prepare


def test_prepare_refuses_to_reuse_non_git_target(tmp_path: Path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(ValueError, match="not a Git clone"):
        prepare(target, Path.cwd(), TraceRecorder("upstream_test", tmp_path / "traces"))
