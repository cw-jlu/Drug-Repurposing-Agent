"""Trace the reproducible v7 PowerPoint build and its source hashes."""

from __future__ import annotations

from pathlib import Path
import subprocess
import zipfile

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


SOURCES = [Path("scripts/build_defense_deck_v7.mjs"),
           Path("benchmark/results/evidence_scope_adjudication_audit.json"),
           Path("benchmark/results/evidence_scope_adjudication_v1.json"),
           Path("benchmark/results/multi_agent_review_v1.json"),
           Path("benchmark/results/recess_official_b3_vs_11.json"),
           Path("benchmark/results/recess_official_b4_vs_11.json"),
           Path("benchmark/results/recess_official_component_ablation.json"),
           Path("benchmark/results/decision_eval_v1.json"),
           Path("benchmark/results/decision_eval_v2.json"),
           Path("benchmark/results/contamination_probe_v1.json"),
           Path("benchmark/results/planner_eval_multistep_v1.json"),
           Path("benchmark/results/agent_v2_demo_runs.json")]
EXPECTED_SLIDES = 12
OUTPUT = Path("deliverables/药物重定位Agent_答辩稿_v7.pptx")


def _main(trace: TraceRecorder) -> None:
    trace.emit("inputs_loaded", inputs={str(path): sha256_file(path) for path in SOURCES},
               output=str(OUTPUT))
    process = subprocess.run(["node", str(SOURCES[0]), str(OUTPUT)], text=True,
                             capture_output=True, encoding="utf-8", errors="replace", check=False)
    trace.emit("builder_finished", exit_code=process.returncode,
               stdout_tail=process.stdout[-1200:], stderr_tail=process.stderr[-1200:])
    if process.returncode:
        raise RuntimeError(f"Deck build failed: {process.stderr[-1200:]}")
    with zipfile.ZipFile(OUTPUT) as archive:
        invalid = archive.testzip()
        slides = len([name for name in archive.namelist()
                      if name.startswith("ppt/slides/slide") and name.endswith(".xml")])
    if invalid or slides != EXPECTED_SLIDES:
        raise ValueError(f"Deck package/layout invalid: {invalid}; slides={slides}")
    trace.emit("deck_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT), slides=slides)
    print(f"{OUTPUT}: {slides} slides; trace: {trace.path}")


if __name__ == "__main__":
    traced_run("defense_deck_v7_build", _main)
