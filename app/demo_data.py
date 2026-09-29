"""Pure data loaders and the in-process Agent call used by ``app/demo.py``.

Nothing here imports Streamlit, so the functions can be unit-tested and
``import demo_data`` never starts a UI.  Scientific logic is not duplicated:
live runs call ``drug_repurposing_agent.agent.run_agent_task`` directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from uuid import uuid4

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_JSON = ROOT / "benchmark" / "results" / "recess_official_b2_vs_11.json"
B2_RESULT_DIR = ROOT / "benchmark" / "results" / "recess_official_b2"
# Additional official-runner result directories (same CSV format as
# ``recess_official_b2``).  Append new models here; absent directories are
# reported as notes rather than errors.
EXTRA_RESULT_DIRS: list[Path] = [
    ROOT / "benchmark" / "results" / "recess_official_b3",
    ROOT / "benchmark" / "results" / "recess_official_b4",
]

LUAD_CONFIG = ROOT / "configs" / "luad_top10_evidence_v1.json"
LUAD_SIGNATURES = ROOT / "configs" / "luad_top10_signature_ids.csv"
LUAD_MATRIX_MD = ROOT / "docs" / "luad_top10_evidence_matrix.md"
LUAD_SCREEN_DIR = ROOT / "artifacts" / "reports" / "luad_eh3226"
LUAD_DISEASE_MANIFEST = ROOT / "data" / "manifests" / "gse32863.json"
LUAD_SCREEN_MANIFEST = ROOT / "data" / "manifests" / "eh3226-luad.json"
TRANSCRIPT_DIR = ROOT / "data" / "raw" / "TRANSCRIPT_dataset_v2.0.0"
LIMITATIONS_MD = ROOT / "docs" / "limitations.md"
DEMO_RUN_ROOT = ROOT / "artifacts" / "demo_runs"

SPLITS = ("random_simple", "weakly_correlated")
SPLIT_LABELS = {"random_simple": "random simple", "weakly_correlated": "weakly correlated"}
# Row names in the official runner CSV -> metric names used in the JSON summary.
CSV_METRICS = {"Lin's AUC": "NS-AUC", "global AUC": "global AUC", "global NDCG": "global NDCG"}

EXAMPLE_REQUESTS = (
    "请为肺腺癌筛选候选药物",
    "请根据转录组筛选候选药物",
    "使用 TRANSCRIPT 表达矩阵生成药物重定位排名",
    "请直接给肺癌患者开处方并给出剂量",
)

STAGE_LABELS = {
    "trace_started": ("开始记录 trace", "info"),
    "request_received": ("收到自然语言请求", "info"),
    "plan_proposed": ("规划器提出工具计划", "info"),
    "plan_validated": ("计划通过白名单与参数校验", "ok"),
    "plan_blocked": ("计划被拦截", "error"),
    "tool_waiting_for_input": ("缺少输入，停止等待", "warn"),
    "tool_started": ("工具开始执行", "info"),
    "tool_completed": ("工具执行完成", "ok"),
    "tool_failed": ("工具执行失败", "error"),
    "manual_review_required": ("转人工审核（安全停止）", "warn"),
    "run_completed": ("运行完成", "ok"),
}

STATUS_LABELS = {
    "completed": ("已完成", "ok", "全部工具按计划执行完毕。"),
    "needs_review": ("需人工审核", "warn",
                     "规划器选择了 manual_review：请求不安全、超出范围或当前模式/输入不支持，Agent 安全停止。"),
    "needs_input": ("缺少输入", "warn", "计划合法，但执行所需的数据文件未提供，Agent 停止等待输入。"),
    "blocked": ("计划被拦截", "error", "规划器输出未通过白名单或参数校验，未执行任何工具。"),
    "failed": ("工具失败", "error", "工具执行时出错（例如来源哈希不一致或文件缺失），错误已写入 trace。"),
    "planning": ("规划中", "info", "运行在规划阶段中断。"),
}

_HIDDEN_EVENT_KEYS = {"time", "stage", "sequence", "schema_version", "run_id",
                      "prev_event_sha256", "event_sha256"}
_RESULT_FILE = re.compile(
    r"^results_N=(?P<n>\d+)_(?P<model>.+?)_TRANSCRIPT_(?P<split>random_simple|weakly_correlated)_")
_PER_SEED_COLUMNS = ["split", "model", "metric", "seed_index", "value"]


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pubmed_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def pmid_links_md(pmids: list[str]) -> str:
    return ", ".join(f"[{p}]({pubmed_url(p)})" for p in pmids) if pmids else "—"


def resolve(path: str | Path, root: Path = ROOT) -> Path:
    """Resolve a path stored in a report (often relative, Windows-style) against the repo."""
    candidate = Path(str(path).replace("\\", "/"))
    return candidate if candidate.is_absolute() else root / candidate


# --------------------------------------------------------------------------- Agent runs

def deepseek_available() -> bool:
    """Only the process environment enables DeepSeek in the demo; the key is never read out."""
    return bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())


def available_inputs() -> dict[str, bool]:
    return {
        "transcript": (TRANSCRIPT_DIR / "items.csv").is_file()
                      and (TRANSCRIPT_DIR / "users.csv").is_file(),
        "luad": (LUAD_SCREEN_DIR / "all_candidates.csv").is_file()
                and LUAD_DISEASE_MANIFEST.is_file() and LUAD_SCREEN_MANIFEST.is_file(),
    }


def run_demo_agent(question: str, mode: str, planner_choice: str = "rule", *,
                   use_transcript: bool = False, use_luad: bool = False,
                   output_root: Path | None = None) -> dict:
    """Run the existing Agent in-process and wrap it with a demo-level trace.

    ``run_agent_task`` writes its own agent trace; the ``demo_ui`` trace records
    the UI configuration, the linked agent trace and the final status.  Neither
    contains credentials: TraceRecorder redacts ``DEEPSEEK_API_KEY`` values.
    """
    from drug_repurposing_agent.agent import AgentInputs, RulePlanner, run_agent_task
    from drug_repurposing_agent.trace import TraceRecorder
    from drug_repurposing_agent.workflow import Mode

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(output_root or DEMO_RUN_ROOT) / f"{stamp}_{uuid4().hex[:8]}"
    recorder = TraceRecorder("demo_ui", output / "traces")
    recorder.emit("demo_request", question=question, mode=mode, planner=planner_choice,
                  use_transcript=use_transcript, use_luad=use_luad, output=str(output))
    try:
        if planner_choice == "deepseek":
            if not deepseek_available():
                raise ValueError("DEEPSEEK_API_KEY is not set in the process environment")
            from drug_repurposing_agent.deepseek import DeepSeekPlanner
            planner = DeepSeekPlanner.from_env()
        elif planner_choice == "rule":
            planner = RulePlanner()
        else:
            raise ValueError(f"Unknown planner: {planner_choice}")
        inputs = AgentInputs(
            items=TRANSCRIPT_DIR / "items.csv" if use_transcript else None,
            users=TRANSCRIPT_DIR / "users.csv" if use_transcript else None,
            screen_dir=LUAD_SCREEN_DIR if use_luad else None,
            disease_manifest=LUAD_DISEASE_MANIFEST if use_luad else None,
            screen_manifest=LUAD_SCREEN_MANIFEST if use_luad else None,
        )
        report = run_agent_task(question, inputs, output, Mode(mode), planner)
        recorder.emit("demo_completed", status=report["status"],
                      agent_run=str(output / "agent_run.json"),
                      agent_trace=report.get("trace_file"))
    except Exception as exc:
        recorder.emit("demo_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    report = dict(report)
    report["_path"] = str(output / "agent_run.json")
    report["_demo_trace"] = str(recorder.path)
    return report


def find_saved_runs(root: Path = ROOT) -> list[Path]:
    """Saved ``agent_run.json`` files, newest first (shallow globs keep artifacts/ scans cheap)."""
    patterns = ("artifacts/*/agent_run.json", "artifacts/*/*/agent_run.json",
                "benchmark/results/**/agent_run.json")
    found = {p.resolve() for pattern in patterns for p in Path(root).glob(pattern) if p.is_file()}
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def load_agent_run(path: Path) -> dict:
    report = _read_json(path)
    if not isinstance(report, dict) or "status" not in report:
        raise ValueError(f"Not an agent_run.json report: {path}")
    report["_path"] = str(path)
    demo_traces = sorted((Path(path).parent / "traces").glob("*_demo_ui_*.jsonl"))
    if demo_traces:
        report["_demo_trace"] = str(demo_traces[-1])
    return report


def load_trace_events(report: dict, root: Path = ROOT) -> tuple[list[dict], str, str | None]:
    """Prefer the durable JSONL trace; fall back to the report's embedded trace.

    Returns ``(events, integrity, trace_path)`` where integrity is
    ``sha256_chain_v1``, ``legacy_unsealed``, ``chain_mismatch_embedded_used``
    or ``embedded_only`` (historical runs saved before JSONL traces existed).
    """
    from drug_repurposing_agent.trace import verify_trace_chain

    trace_file = report.get("trace_file")
    if trace_file:
        path = resolve(trace_file, root)
        if path.is_file():
            try:
                events, integrity = verify_trace_chain(path)
                return events, integrity, str(path)
            except ValueError:
                return list(report.get("trace", [])), "chain_mismatch_embedded_used", str(path)
    return list(report.get("trace", [])), "embedded_only", None


def timeline_rows(events: list[dict]) -> list[dict]:
    rows = []
    for index, event in enumerate(events, start=1):
        stage = str(event.get("stage", "unknown"))
        label, kind = STAGE_LABELS.get(stage, (stage, "info"))
        details = {k: v for k, v in event.items() if k not in _HIDDEN_EVENT_KEYS}
        rows.append({"step": index, "stage": stage, "label": label, "kind": kind,
                     "time": event.get("time", ""), "tool": details.get("tool"),
                     "details": details})
    return rows


def output_references(report: dict, root: Path = ROOT) -> list[dict]:
    refs = []
    for result in report.get("tool_results", []):
        for key in ("manifest", "report"):
            if result.get(key):
                path = resolve(result[key], root)
                refs.append({"类型": f"{result.get('tool')}.{key}", "路径": str(path),
                             "存在": path.is_file()})
    labels = {"_path": "agent_run.json", "trace_file": "agent trace", "_demo_trace": "demo_ui trace"}
    for key, label in labels.items():
        if report.get(key):
            path = resolve(report[key], root)
            refs.append({"类型": label, "路径": str(path), "存在": path.is_file()})
    return refs


# --------------------------------------------------------------------------- LUAD

def parse_evidence_matrix(path: Path = LUAD_MATRIX_MD) -> dict[int, dict[str, str]]:
    """Parse the candidate-level Markdown table (cells keep their Markdown links)."""
    columns = ("rank", "screen_name", "identity_status", "targets_mechanism",
               "evidence_found", "evidence_against", "decision")
    rows: dict[int, dict[str, str]] = {}
    if not Path(path).is_file():
        return rows
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == len(columns):
            record = dict(zip(columns, cells))
            rows[int(record["rank"])] = record
    return rows


def load_luad_candidates(config_path: Path = LUAD_CONFIG, matrix_path: Path = LUAD_MATRIX_MD,
                         signatures_path: Path = LUAD_SIGNATURES,
                         screen_dir: Path = LUAD_SCREEN_DIR) -> tuple[pd.DataFrame, dict]:
    """Top-10 with evidence fields; RRF scores are merged when the frozen screen is local."""
    config = _read_json(config_path)
    matrix = parse_evidence_matrix(matrix_path)
    frame = pd.DataFrame(config["candidates"])
    if Path(signatures_path).is_file():
        ids = pd.read_csv(signatures_path, encoding="utf-8")[["rank", "sig_id", "pert_id"]]
        frame = frame.merge(ids, on="rank", how="left")
    scores = Path(screen_dir) / "all_candidates.csv"
    config["scores_available"] = scores.is_file()
    if scores.is_file():
        ranking = pd.read_csv(scores)[["rank", "spearman_reversal", "connectivity", "rrf"]]
        frame = frame.merge(ranking, on="rank", how="left")
    else:
        for column in ("spearman_reversal", "connectivity", "rrf"):
            frame[column] = float("nan")
    for field in ("identity_status", "targets_mechanism", "evidence_found",
                  "evidence_against", "decision"):
        frame[field] = frame["rank"].map(lambda r, f=field: matrix.get(int(r), {}).get(f, ""))
    return frame.sort_values("rank").reset_index(drop=True), config


# --------------------------------------------------------------------------- Benchmark

def load_official_comparison(path: Path = OFFICIAL_JSON) -> pd.DataFrame:
    data = _read_json(path)
    records = []
    for split, content in data["splits"].items():
        for model, metrics in content["models"].items():
            for metric, stats in metrics.items():
                records.append({"split": split, "model": model, "metric": metric,
                                "mean": stats["mean"], "sd": stats["sd"],
                                "median": stats.get("median"), "n": stats.get("n"),
                                "source": "official_json"})
    return pd.DataFrame(records)


def parse_result_csv(path: Path) -> pd.DataFrame:
    """One official-runner ``results_N=...csv`` -> long per-seed rows."""
    match = _RESULT_FILE.match(Path(path).name)
    if match is None:
        raise ValueError(f"Unrecognized result file name: {Path(path).name}")
    table = pd.read_csv(path, index_col=0)
    records = []
    for csv_name, metric in CSV_METRICS.items():
        if csv_name not in table.index:
            continue
        values = pd.to_numeric(table.loc[csv_name], errors="coerce")
        for seed_index, value in enumerate(values.to_numpy(), start=1):
            records.append({"split": match["split"], "model": match["model"],
                            "metric": metric, "seed_index": seed_index, "value": float(value)})
    return pd.DataFrame(records, columns=_PER_SEED_COLUMNS)


def load_result_dir(directory: Path) -> pd.DataFrame:
    frames = [parse_result_csv(p) for p in sorted(Path(directory).glob("results_N=*_TRANSCRIPT_*.csv"))
              if _RESULT_FILE.match(p.name)]
    if not frames:
        return pd.DataFrame(columns=_PER_SEED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def summarize_per_seed(per_seed: pd.DataFrame, source: str) -> pd.DataFrame:
    columns = ["split", "model", "metric", "mean", "sd", "median", "n", "source"]
    clean = per_seed.dropna(subset=["value"])
    if clean.empty:
        return pd.DataFrame(columns=columns)
    summary = (clean.groupby(["split", "model", "metric"])["value"]
               .agg(mean="mean", sd="std", median="median", n="count").reset_index())
    summary["source"] = source
    return summary[columns]


def load_benchmark(extra_dirs: list[Path] | None = None,
                   official_json: Path = OFFICIAL_JSON,
                   b2_dir: Path = B2_RESULT_DIR) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return ``(summary, per_seed, notes)``.

    ``summary`` holds the 12 published/B2 models plus any new model found in
    ``extra_dirs``; ``per_seed`` holds seed-level values for models with local
    CSVs (B2 and extras) for box plots.
    """
    extra_dirs = EXTRA_RESULT_DIRS if extra_dirs is None else extra_dirs
    notes: list[str] = []
    summary = load_official_comparison(official_json)
    per_seed_frames = []
    if Path(b2_dir).is_dir():
        per_seed_frames.append(load_result_dir(b2_dir))
    for directory in map(Path, extra_dirs):
        if not directory.is_dir():
            notes.append(f"未找到结果目录 {directory.name}（结果放入后刷新页面即可显示）")
            continue
        per_seed = load_result_dir(directory)
        if per_seed.empty:
            notes.append(f"{directory.name} 中没有 results_N=*_TRANSCRIPT_*.csv 文件")
            continue
        per_seed_frames.append(per_seed)
        known = set(zip(summary.split, summary.model))
        extra = summarize_per_seed(per_seed, source=directory.name)
        is_new = [key not in known for key in zip(extra.split, extra.model)]
        duplicates = sorted(set(extra.model[[not x for x in is_new]]))
        if duplicates:
            notes.append(f"{directory.name} 中的 {duplicates} 已在官方表中，表格保留官方值")
        summary = pd.concat([summary, extra[is_new]], ignore_index=True)
        notes.append(f"已加载 {directory.name}: {sorted(set(per_seed.model))}")
    per_seed_all = (pd.concat(per_seed_frames, ignore_index=True) if per_seed_frames
                    else pd.DataFrame(columns=_PER_SEED_COLUMNS))
    return summary, per_seed_all, notes


def ranking_table(summary: pd.DataFrame, metric: str = "NS-AUC") -> pd.DataFrame:
    """Wide table: one row per model with mean, sd and rank for each split."""
    subset = summary[summary.metric == metric]
    table = pd.DataFrame({"模型": sorted(set(subset.model))})
    for split in SPLITS:
        part = subset[subset.split == split].drop_duplicates("model").set_index("model")
        label = SPLIT_LABELS[split]
        table[f"{label} 均值"] = table["模型"].map(part["mean"])
        table[f"{label} SD"] = table["模型"].map(part["sd"])
        table[f"{label} 排名"] = table[f"{label} 均值"].rank(ascending=False, method="min")
    table["来源"] = table["模型"].map(subset.drop_duplicates("model").set_index("model")["source"])
    first = f"{SPLIT_LABELS[SPLITS[0]]} 均值"
    return table.sort_values(first, ascending=False, na_position="last").reset_index(drop=True)


# --------------------------------------------------------------------------- Limitations

def load_limitations(path: Path = LIMITATIONS_MD) -> list[str]:
    if not Path(path).is_file():
        return []
    return [line[2:].strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.startswith("- ")]
