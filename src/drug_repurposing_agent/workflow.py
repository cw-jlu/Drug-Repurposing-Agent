"""Small state machine with an explicit benchmark data boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .data import ExpressionData, sha256_file
from .ranking import score_expressions


class Mode(str, Enum):
    BENCHMARK_STRICT = "benchmark_strict"
    RESEARCH_OPEN = "research_open"


@dataclass
class RunState:
    mode: Mode
    run_id: str = field(default_factory=lambda: uuid4().hex)
    stage: str = "created"
    trace: list[dict] = field(default_factory=list)

    def step(self, stage: str, **details: object) -> None:
        self.stage = stage
        self.trace.append({"time": datetime.now(timezone.utc).isoformat(),
                           "stage": stage, **details})


def validate_scores(scores: dict[str, pd.DataFrame], data: ExpressionData) -> None:
    expected = (list(data.drugs.columns), list(data.diseases.columns))
    for name, matrix in scores.items():
        if list(matrix.index) != expected[0] or list(matrix.columns) != expected[1]:
            raise ValueError(f"{name}: matrix IDs or order differ from input")
        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: non-finite score")
    if (scores["gene_counts"].to_numpy() < 0).any():
        raise ValueError("Negative gene overlap")


def run_expression_workflow(items: Path, users: Path, output: Path,
                            mode: Mode = Mode.BENCHMARK_STRICT,
                            top_k: int = 100) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    state = RunState(mode)
    state.step("plan", allowed_sources=["items", "users"] if mode == Mode.BENCHMARK_STRICT
               else ["items", "users", "curated_evidence"])
    data = ExpressionData.from_csv(items, users)
    qc = data.qc()
    state.step("quality_control", **qc)
    scores = score_expressions(data, top_k=top_k)
    validate_scores(scores, data)
    state.step("ranked", methods=["spearman_reversal", "connectivity", "rrf"])
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in scores.items():
        frame.to_csv(output / f"{name}.csv", index_label="drug_id")
    manifest = {
        "run_id": state.run_id, "mode": mode.value, "status": "completed",
        "methods": {"connectivity_top_k": top_k, "rrf_k": 60,
                    "missing_score": 0, "undefined_correlation": 0},
        "input": {"items": {"path": str(items), "sha256": sha256_file(items)},
                  "users": {"path": str(users), "sha256": sha256_file(users)}},
        "qc": qc, "outputs": {name: f"{name}.csv" for name in scores},
        "trace": state.trace,
        "limitations": ["Transcriptomic reversal is not clinical efficacy.",
                        "Pathway and external evidence are unavailable in benchmark strict mode."],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
