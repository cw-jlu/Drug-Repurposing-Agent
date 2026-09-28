"""Build the defense figures (Benchmark + LUAD biological interpretation).

Run from the repository root:

    python scripts/make_figures.py [--published <RECeSS benchmark-results checkout>]

Writes PNG (dpi 200) and SVG files to ``docs/figures/``. Every input is read
from versioned project files, the pinned published RECeSS results, or raw data
already under ``data/raw/``; no figure value is typed in by hand. A trace is
written to ``artifacts/traces/`` as required by AGENTS.md.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from drug_repurposing_agent.trace import TraceRecorder  # noqa: E402

FIG_DIR = REPO / "docs" / "figures"
SPLITS = {"random_simple": "Random simple split", "weakly_correlated": "Weakly correlated split"}
PUBLISHED_SPLIT_DIRS = {"random_simple": "results_TRANSCRIPT",
                        "weakly_correlated": "results_TRANSCRIPT_weakly_correlated"}
PUBLISHED_MODELS = ["ALSWR", "BNNR", "DDA_SKF", "FastaiCollabWrapper", "HAN", "LibMF",
                    "LogisticMF", "MBiRW", "NIMCGCN", "PMF", "SCPMF"]
# Our methods: add a new entry when a teammate's run lands (same CSV format).
# Missing directories are skipped.
OUR_RESULT_DIRS = {"B2": "benchmark/results/recess_official_b2",
                   "B3": "benchmark/results/recess_official_b3"}
METRIC_ROW = "Lin's AUC"  # = NS-AUC in the RECeSS analysis

# Palette: dataviz reference palette, checked with validate_palette.js.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a",
                                           "#eda100", "#4a3aa7", "#e34948")
PUBLISHED_COLOR = "#c3c2b7"
OUR_COLORS = [ORANGE, BLUE, AQUA, VIOLET]


def setup_fonts() -> None:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                             "Source Han Sans SC", "PingFang SC") if name in available]
    if not cjk:
        raise SystemExit("No CJK font found; install Microsoft YaHei/SimHei/Noto Sans CJK.")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": cjk + ["DejaVu Sans"],
        "axes.unicode_minus": False, "svg.fonttype": "path",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11.5,
        "figure.facecolor": "white", "axes.facecolor": "white"})


def save(fig: plt.Figure, name: str, trace: TraceRecorder, **details: object) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    trace.emit("figure_written", figure=name, **details)
    print(f"wrote docs/figures/{name}.png/.svg")


# ---------------------------------------------------------------- fig1 -----
def _result_file(folder: Path, model: str, split: str) -> Path | None:
    hits = sorted(folder.glob(f"results_N=*_{model}_TRANSCRIPT_{split}_AUC_1.000000_5_0.200000.csv"))
    return hits[0] if hits else None


def _seeds(result: Path) -> list[int] | None:
    seeds = result.with_name(result.name.replace("results_", "seeds_", 1))
    if not seeds.exists():
        return None
    return pd.read_csv(seeds, index_col=0).loc["seed"].astype(int).tolist()


def _ns_auc(result: Path) -> np.ndarray:
    values = pd.read_csv(result, index_col=0).loc[METRIC_ROW].astype(float).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite NS-AUC in {result}")
    return values


def load_fig1(published: Path) -> tuple[dict, dict]:
    data, notes = {}, {}
    for split in SPLITS:
        ours = {}
        for model, rel in OUR_RESULT_DIRS.items():
            path = _result_file(REPO / rel, model, split)
            if path is None:
                print(f"fig1: skip {model}/{split} (no results under {rel})")
                continue
            ours[model] = (_ns_auc(path), _seeds(path))
        n = min([len(v) for v, _ in ours.values()] or [100])
        seed_lists = [s[:n] for _, s in ours.values() if s]
        if any(s != seed_lists[0] for s in seed_lists):
            raise ValueError(f"Seed order differs among our models for {split}")
        reference = seed_lists[0] if seed_lists else None
        split_data = {}
        for model in PUBLISHED_MODELS:
            path = _result_file(published / PUBLISHED_SPLIT_DIRS[split] / f"results_{model}", model, split)
            if path is None:
                raise FileNotFoundError(f"Missing published {model} {split} under {published}")
            values, seeds = _ns_auc(path), _seeds(path)
            if reference and seeds and seeds[:n] != reference:
                raise ValueError(f"Seed order mismatch for {model}/{split}")
            split_data[model] = values[:n]
        for model, (values, _) in ours.items():
            split_data[model] = values[:n]
        data[split] = split_data
        notes[split] = n
    return data, notes


def fig1(published: Path, trace: TraceRecorder) -> dict:
    data, notes = load_fig1(published)
    ours = [m for m in OUR_RESULT_DIRS if m in data["random_simple"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharex=True)
    medians = {}
    for ax, (split, title) in zip(axes, SPLITS.items()):
        models = sorted(data[split], key=lambda m: np.median(data[split][m]))
        medians[split] = {m: round(float(np.median(data[split][m])), 4) for m in models[::-1]}
        colors = [OUR_COLORS[ours.index(m)] if m in ours else PUBLISHED_COLOR for m in models]
        bp = ax.boxplot([data[split][m] for m in models], vert=False, widths=0.62,
                        patch_artist=True, showfliers=True,
                        medianprops={"color": INK, "linewidth": 1.6},
                        whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": MUTED,
                                    "markeredgecolor": "none", "alpha": .6})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("white" if color == PUBLISHED_COLOR else INK)
            patch.set_linewidth(0.8)
        ax.axvline(0.5, color=INK2, linestyle="--", linewidth=1)
        ax.set_yticks(range(1, len(models) + 1))
        ax.set_yticklabels(models)
        for label, m in zip(ax.get_yticklabels(), models):
            if m in ours:
                label.set_fontweight("bold")
                label.set_color(INK)
        for i, m in enumerate(models, start=1):
            ax.text(1.01, i, f"{np.median(data[split][m]):.3f}", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=8.5, color=INK if m in ours else INK2,
                    fontweight="bold" if m in ours else "normal")
        ax.text(1.01, 1.0, "中位数", transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")
        n = notes[split]
        suffix = "" if n == 100 else f"（仅比较前 {n} 个公开种子）"
        ax.set_title(f"{title}  (n={n} seeds){suffix}", loc="left")
        ax.set_xlabel("NS-AUC（行平均 AUC，官方 “Lin's AUC”）")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    handles = [Patch(facecolor=PUBLISHED_COLOR, label="RECeSS 公开 11 模型")] + \
              [Patch(facecolor=OUR_COLORS[i], edgecolor=INK, label=f"本项目 {m}") for i, m in enumerate(ours)] + \
              [plt.Line2D([], [], color=INK2, linestyle="--", label="0.5 参考线")]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.005, 0.985))
    fig.suptitle("TRANSCRIPT 官方 RECeSS 流程：每个种子的 NS-AUC 分布（按中位数排序）",
                 x=0.01, ha="left", y=1.03, fontsize=13, color=INK)
    fig.subplots_adjust(wspace=0.62, top=0.86)
    save(fig, "fig1_nsauc_boxplot", trace, medians=medians, seeds=notes, our_models=ours)
    return medians


# ---------------------------------------------------------------- fig2 -----
def fig2(trace: TraceRecorder) -> dict:
    src = REPO / "benchmark/results/recess_official_component_ablation.json"
    report = json.loads(src.read_text(encoding="utf-8"))
    models = ["B0p", "B1k", "B1", "B2"]
    labels = {"B0p": "B0p 训练集药物流行度", "B1k": "B1k 表达近邻标签传播",
              "B1": "B1 无标签表达逆转", "B2": "B2 固定 RRF 融合"}
    colors = [BLUE, YELLOW, AQUA, ORANGE]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    width, gap = 0.19, 0.012
    values = {}
    for j, split in enumerate(SPLITS):
        for i, m in enumerate(models):
            stats = report["splits"][split]["models"][m]["NS-AUC"]
            values[f"{split}/{m}"] = round(stats["mean"], 4)
            x = j + (i - 1.5) * (width + gap)
            ax.bar(x, stats["mean"], width, color=colors[i], yerr=stats["sd"],
                   error_kw={"ecolor": INK2, "elinewidth": 0.9, "capsize": 3},
                   label=labels[m] if j == 0 else None)
            ax.text(x, stats["mean"] + stats["sd"] + 0.012, f"{stats['mean']:.3f}",
                    ha="center", va="bottom", fontsize=8.8, color=INK)
    ax.axhline(0.5, color=INK2, linestyle="--", linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random simple", "Weakly correlated"], fontsize=10.5)
    ax.set_ylim(0, 0.98)
    ax.set_yticks(np.arange(0, 0.81, 0.1))
    ax.set_ylabel("NS-AUC（100 种子均值 ± SD）")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.085), fontsize=9)
    ax.set_title("B2 组件消融（同一官方流程、100 种子、五折选模）", loc="left", pad=30)
    note = ("B0p = 0.5 是退化回落值，不代表“随机水平”的排序能力：药物流行度在同一药物行内为常数，\n"
            "行内正/非正样本全部并列；官方行式指标用严格 “>” 比较，并列得 0 分，该行无区分度，结果回落为 0.5。\n"
            "B1k 远低于 0.5 也主要来自大量并列分数（严格比较下并列计 0），而非分数方向相反（见 b1k_metric_audit.md）。")
    ax.text(0.01, 0.975, note, transform=ax.transAxes, ha="left", va="top", fontsize=8.3,
            color=INK2, bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f6f5f2",
                              "edgecolor": GRID})
    save(fig, "fig2_component_ablation", trace, source=str(src.relative_to(REPO)), values=values)
    return values


# ---------------------------------------------------------------- fig3 -----
def load_deg(trace: TraceRecorder) -> tuple[pd.DataFrame, str]:
    processed = REPO / "data/processed/luad/luad_deg_all.tsv"
    if processed.exists():
        return pd.read_csv(processed, sep="\t"), str(processed.relative_to(REPO))
    # Recompute in memory with the functions of scripts/process_gse32863.py.
    # Nothing is written; the versioned manifest is left untouched.
    from drug_repurposing_agent.differential_expression import paired_deg
    from scripts import process_gse32863 as proc
    series, annot = REPO / proc.SERIES, REPO / proc.ANNOTATION
    series_start, metadata = proc.table_start(series, "!series_matrix_table_begin")
    samples = proc.sample_manifest(metadata)
    matrix = pd.read_csv(series, sep="\t", skiprows=series_start, nrows=48803,
                         index_col=0, compression="gzip")
    annotation_start, _ = proc.table_start(annot, "!platform_table_begin")
    annotation = pd.read_csv(annot, sep="\t", skiprows=annotation_start, compression="gzip",
                             usecols=["ID", "Gene symbol"], dtype=str,
                             comment="!").drop_duplicates("ID").set_index("ID")
    mapped = matrix.join(annotation, how="left")
    valid = mapped["Gene symbol"].fillna("").str.fullmatch(r"[A-Za-z][A-Za-z0-9.\-]*")
    mapped = mapped.loc[valid].copy()
    values = mapped[matrix.columns]
    mapped["probe_iqr"] = values.quantile(0.75, axis=1) - values.quantile(0.25, axis=1)
    mapped["probe_id"] = mapped.index
    selected = (mapped.sort_values(["Gene symbol", "probe_iqr", "probe_id"],
                                   ascending=[True, False, True])
                .drop_duplicates("Gene symbol").set_index("Gene symbol"))
    included = samples.loc[samples.included, "sample_id"]
    deg = paired_deg(selected[matrix.columns][included], samples.loc[samples.included])
    manifest = json.loads((REPO / "data/manifests/gse32863.json").read_text(encoding="utf-8"))
    expected = manifest["outputs"]["luad_deg_all.tsv"]["sha256"]
    text = deg.to_csv(sep="\t", index=False, lineterminator="\n")
    match = expected in {sha256(text.encode()).hexdigest(),
                         sha256(text.replace("\n", "\r\n").encode()).hexdigest()}
    trace.emit("deg_recomputed", genes=len(deg), sha256_matches_manifest=match)
    print(f"fig3: DEG recomputed in memory from data/raw/GSE32863; sha256 matches manifest: {match}")
    return deg, "data/raw/GSE32863 (in-memory rerun of process_gse32863 logic)"


def fig3(trace: TraceRecorder) -> dict:
    deg, source = load_deg(trace)
    y = -np.log10(deg["fdr"].clip(lower=1e-300))
    up = deg.included_default & (deg.direction == "up")
    down = deg.included_default & (deg.direction == "down")
    ns = ~(up | down)
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.scatter(deg.log2FC[ns], y[ns], s=4, color="#d4d2cb", linewidths=0, rasterized=True,
               label=f"不显著 ({int(ns.sum()):,})")
    ax.scatter(deg.log2FC[up], y[up], s=7, color=RED, linewidths=0, rasterized=True,
               label=f"上调 ({int(up.sum())})")
    ax.scatter(deg.log2FC[down], y[down], s=7, color=BLUE, linewidths=0, rasterized=True,
               label=f"下调 ({int(down.sum())})")
    ax.axhline(-np.log10(0.05), color=MUTED, linestyle="--", linewidth=0.9)
    for v in (-1, 1):
        ax.axvline(v, color=MUTED, linestyle="--", linewidth=0.9)
    xmax = float(np.ceil(deg.log2FC.abs().max() + 1.5))
    ymax = float(y.max()) * 1.05
    labeled = []
    # Top 8 genes per direction by |log2FC|, placed in an evenly spaced column
    # beside the cloud so labels never overlap.
    for mask, side in ((up, 1), (down, -1)):
        sub = deg[mask].assign(y=y[mask])
        top = sub.loc[sub.log2FC.abs().sort_values(ascending=False).index].head(8).sort_values("y")
        # Slots follow the labeled points' own y-range (min spacing 4.5% of
        # the axis) so leader lines stay short and rarely cross.
        span = max(top.y.max() - top.y.min(), ymax * 0.045 * (len(top) - 1))
        lo = min(max(top.y.mean() - span / 2, ymax * 0.05), ymax * 0.95 - span)
        slots = np.linspace(lo, lo + span, len(top))
        tx = side * (xmax - 0.2)
        for row, ty in zip(top.itertuples(), slots):
            ax.annotate(row.gene_symbol, (row.log2FC, row.y), xytext=(tx, ty),
                        fontsize=8, color=INK, ha="right" if side > 0 else "left", va="center",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5,
                                    "shrinkA": 1, "shrinkB": 2})
            labeled.append(row.gene_symbol)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("log2FC（肿瘤 − 配对正常）")
    ax.set_ylabel("−log10(BH FDR)")
    ax.set_title(f"GSE32863 LUAD 配对差异表达（57 对，{len(deg):,} 基因，配对 t 检验）\n"
                 f"阈值 FDR<0.05 且 |log2FC|≥1：上调 {int(up.sum())}、下调 {int(down.sum())}"
                 "（标注各方向 |log2FC| 最大的 8 个基因）", loc="left")
    ax.legend(frameon=False, loc="lower right", markerscale=2.5, fontsize=9)
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, "fig3_luad_volcano", trace, source=source, up=int(up.sum()), down=int(down.sum()),
         labeled=labeled)
    return {"up": int(up.sum()), "down": int(down.sum()), "genes": len(deg), "source": source,
            "labeled": labeled}


# ---------------------------------------------------------------- fig4 -----
IDENTITY_TIERS = [  # (regex on the recorded identity string, label, color)
    ("exact_Hub_InChIKey", "来源 ID 精确 + Hub InChIKey 一致", AQUA),
    ("Hub_stereochemistry_differs", "来源 ID 精确；Hub 立体化学不同", BLUE),
    ("mismatch|ambiguous", "来源 ID 精确；Hub 结构不符/名称歧义", ORANGE),
    ("no_Hub_sample", "来源 ID 精确；Hub 无样品", VIOLET),
]


def identity_tier(identity: str) -> tuple[str, str]:
    for pattern, label, color in IDENTITY_TIERS:
        if re.search(pattern, identity):
            return label, color
    return "未分类", MUTED


def fig4(trace: TraceRecorder) -> dict:
    config = json.loads((REPO / "configs/luad_top10_evidence_v1.json").read_text(encoding="utf-8"))
    cands = sorted(config["candidates"], key=lambda c: c["rank"])
    scores_path = REPO / "artifacts/reports/luad_eh3226/all_candidates.csv"
    scores = pd.read_csv(scores_path).set_index("drug_name") if scores_path.exists() else None
    rows = cands[::-1]  # rank 1 on top
    tiers = [identity_tier(c["identity"]) for c in rows]
    ypos = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    if scores is not None:
        vals = [float(scores.loc[c["name"], "rrf"]) for c in rows]
        ax.barh(ypos, vals, color=[t[1] for t in tiers], height=0.62)
        for yv, v in zip(ypos, vals):
            ax.text(v, yv, f" {v:.4f}", va="center", fontsize=8.5, color=INK)
        ax.set_xlabel("RRF 融合分数（负 Spearman + 上/下调基因集连接性，k=60）")
        ax.legend(handles=[Patch(facecolor=c, label=l) for _, l, c in IDENTITY_TIERS],
                  frameon=False, fontsize=8.5, loc="lower right")
        mode = "rrf_scores"
        subtitle = ""
    else:
        # RRF scores need the 2.46 GB EH3226 file; show only what is recorded.
        ax.set_xlim(0, 10)
        for yv, c, (label, color) in zip(ypos, rows, tiers):
            ax.add_patch(plt.Rectangle((0, yv - 0.3), 0.3, 0.6, color=color))
            ax.text(0.45, yv, label, va="center", fontsize=9, color=INK)
            ax.text(5.3, yv, str(len(c["targets"])), va="center", ha="center", fontsize=9.5, color=INK)
            ax.text(6.7, yv, str(len(c["candidate_pmids"])), va="center", ha="center",
                    fontsize=9.5, color=INK)
            ax.text(7.6, yv, c["confidence"], va="center", fontsize=9, color=RED)
        for x, head, ha in ((0.0, "身份核查分层", "left"), (5.3, "记录靶点数", "center"),
                            (6.7, "候选特异 PMID", "center"), (7.6, "证据结论", "left")):
            ax.text(x, len(rows) - 0.35, head, fontsize=9.5, color=INK2, fontweight="bold", ha=ha)
        for yv in ypos[:-1]:
            ax.axhline(yv + 0.5, color=GRID, linewidth=0.6)
        ax.set_ylim(-0.6, len(rows) + 0.05)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        mode = "evidence_table_no_scores"
        subtitle = ("\nRRF 分数文件 all_candidates.csv 未生成（需 2.46 GB EH3226 重跑），"
                    "此处按排名列出已记录证据")
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"#{c['rank']}  {c['name']}" for c in rows], fontsize=9.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("LUAD 表达逆转 Top-10（A549，10 µM，24 h）：9/10 为糖皮质激素类，10/10 证据不足"
                 + subtitle, loc="left", fontsize=11)
    save(fig, "fig4_luad_top10", trace, mode=mode)
    return {"mode": mode}


# ---------------------------------------------------------------- fig5 -----
PATHWAY_LABELS = {
    "glucocorticoid_receptor_transcription": "GR 转录调控",
    "annexin_eicosanoid_signaling": "Annexin/类花生酸信号",
    "xenobiotic_metabolism": "外源物代谢",
    "phospholipase_eicosanoid_context": "磷脂酶/类花生酸",
    "parent_drug_glucocorticoid_receptor_inference_only": "GR（仅母药推断）",
}


def fig5(trace: TraceRecorder) -> dict:
    import networkx as nx
    config = json.loads((REPO / "configs/luad_top10_evidence_v1.json").read_text(encoding="utf-8"))
    cands = sorted(config["candidates"], key=lambda c: c["rank"])
    g = nx.Graph()
    for c in cands:
        d = f"#{c['rank']} {c['name']}"
        g.add_node(d, kind="drug")
        for t in c["targets"]:
            g.add_node(t, kind="target")
            g.add_edge(d, t, kind="target")
        for p in c["pathways"]:
            g.add_node(p, kind="pathway")
            g.add_edge(d, p, kind="pathway", inferred="inference" in p)
    drugs = [n for n, a in g.nodes(data=True) if a["kind"] == "drug"]
    targets = sorted([n for n, a in g.nodes(data=True) if a["kind"] == "target"],
                     key=lambda t: (-g.degree(t), t))
    pathways = sorted([n for n, a in g.nodes(data=True) if a["kind"] == "pathway"],
                      key=lambda p: (-g.degree(p), p))

    def column(nodes, x, lo=0.0, hi=1.0):
        ys = np.linspace(hi, lo, len(nodes)) if len(nodes) > 1 else [(lo + hi) / 2]
        return {n: (x, y) for n, y in zip(nodes, ys)}

    pos = {**column(targets, 0.0, 0.05, 0.95), **column(drugs, 1.0),
           **column(pathways, 2.0, 0.2, 0.8)}
    fig, ax = plt.subplots(figsize=(12, 7))
    for u, v, a in g.edges(data=True):
        (x1, y1), (x2, y2) = pos[u], pos[v]
        nr3c1 = "NR3C1" in (u, v)
        gr = "glucocorticoid_receptor_transcription" in (u, v)
        color = ORANGE if nr3c1 else (AQUA if gr else MUTED)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.6 if (nr3c1 or gr) else 0.9,
                alpha=0.75, linestyle=":" if a.get("inferred") else "-", zorder=1)
    for n in drugs:
        x, y = pos[n]
        isolated = g.degree(n) == 0
        ax.scatter([x], [y], s=90, color="white" if isolated else INK2, edgecolor=INK2,
                   linewidth=1.2, zorder=3)
        ax.text(x, y + 0.028, n + ("（无记录靶点/通路）" if isolated else ""), ha="center",
                va="bottom", fontsize=8.6, color=INK, zorder=4,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3, "alpha": 0.9})
    for n in targets:
        x, y = pos[n]
        ax.scatter([x], [y], s=50 + 30 * g.degree(n), color=ORANGE if n == "NR3C1" else BLUE,
                   edgecolor="white", linewidth=1, zorder=3)
        ax.text(x - 0.06, y, f"{n}（{g.degree(n)}）", ha="right", va="center", fontsize=9, color=INK)
    for n in pathways:
        x, y = pos[n]
        ax.scatter([x], [y], s=50 + 30 * g.degree(n), marker="s", color=AQUA,
                   edgecolor="white", linewidth=1, zorder=3)
        ax.text(x + 0.06, y, f"{PATHWAY_LABELS.get(n, n)}（{g.degree(n)}）", ha="left",
                va="center", fontsize=9, color=INK)
    for x, head in ((0.0, "记录靶点（Broad 名称级注释）"), (1.0, "Top-10 候选"), (2.0, "记录通路")):
        ax.text(x, 1.09, head, ha="center", fontsize=10.5, fontweight="bold", color=INK2)
    ax.set_xlim(-0.75, 2.9)
    ax.set_ylim(-0.05, 1.13)
    ax.axis("off")
    nr = g.degree("NR3C1") if "NR3C1" in g else 0
    gr_key = "glucocorticoid_receptor_transcription"
    ngr = g.degree(gr_key) if gr_key in g else 0
    ax.set_title(f"Top-10 药物–靶点–通路网络：{nr}/10 记录靶点含 NR3C1，{ngr}/10 记录 GR 转录通路——同一机制轴\n"
                 "仅使用 configs/luad_top10_evidence_v1.json 已记录字段；点线 = 仅母药推断；括号 = 连接候选数",
                 loc="left", fontsize=11)
    save(fig, "fig5_top10_network", trace, nodes=g.number_of_nodes(), edges=g.number_of_edges(),
         nr3c1_degree=nr, gr_pathway_degree=ngr)
    return {"nr3c1": nr, "gr_pathway": ngr, "targets": targets, "pathways": pathways}


# ---------------------------------------------------------------- fig6 -----
def fig6(trace: TraceRecorder) -> dict:
    src = REPO / "data/raw/TRANSCRIPT_dataset_v2.0.0/ratings_mat.csv"
    v = pd.read_csv(src, index_col=0).to_numpy()
    n_pos, n_neg = int((v == 1).sum()), int((v == -1).sum())
    density = (n_pos + n_neg) / v.size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.05]})
    pi, pj = np.nonzero(v == 1)
    ni, nj = np.nonzero(v == -1)
    ax1.scatter(pj, pi, s=6, marker="s", color=BLUE, linewidths=0, label=f"+1 已知有效 ({n_pos})")
    ax1.scatter(nj, ni, s=22, marker="x", color=RED, linewidths=1.2, label=f"−1 已知失败 ({n_neg})")
    ax1.set_xlim(-1, v.shape[1])
    ax1.set_ylim(v.shape[0], -1)
    ax1.set_xlabel(f"疾病（{v.shape[1]} 列）")
    ax1.set_ylabel(f"药物（{v.shape[0]} 行）")
    ax1.set_title(f"关联矩阵 {v.shape[0]}×{v.shape[1]}：已知标签仅占 {density:.2%}", loc="left")
    ax1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2,
               fontsize=8.5, markerscale=1.5)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(True)
    per_disease = (v == 1).sum(axis=0)
    per_drug = (v == 1).sum(axis=1)
    bins = np.arange(0, per_disease.max() + 2) - 0.5
    ax2.hist(per_disease, bins=bins, color=BLUE, edgecolor="white", linewidth=0.8)
    ax2.set_xlabel("每种疾病的阳性（+1）药物数")
    ax2.set_ylabel("疾病数")
    single, zero = int((per_disease == 1).sum()), int((per_disease == 0).sum())
    ax2.set_title(f"阳性长尾：中位数 {np.median(per_disease):.0f}，{single}/{v.shape[1]} 种疾病仅 1 个阳性",
                  loc="left")
    ax2.grid(axis="y", color=GRID, linewidth=0.6)
    ax2.set_axisbelow(True)
    ax2.text(0.97, 0.95, f"0 个阳性的疾病：{zero}\n至少 1 个阳性的药物：{int((per_drug > 0).sum())}/{v.shape[0]}\n"
             f"单病最多阳性：{per_disease.max()}", transform=ax2.transAxes, ha="right", va="top",
             fontsize=9, color=INK2)
    fig.suptitle("TRANSCRIPT v2.0.0 标签概况：极度稀疏且长尾", x=0.01, ha="left", fontsize=12.5,
                 color=INK, y=1.0)
    fig.subplots_adjust(wspace=0.25)
    stats = {"shape": list(v.shape), "pos": n_pos, "neg": n_neg, "density": round(density, 5),
             "median_pos_per_disease": float(np.median(per_disease)), "single": single,
             "zero": zero, "max": int(per_disease.max()), "drugs_with_pos": int((per_drug > 0).sum())}
    save(fig, "fig6_data_overview", trace, **stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--published", type=Path,
                        default=Path(os.environ.get("TEMP", "/tmp")) / "recess-benchmark-results-audit-20260925",
                        help="Checkout of RECeSS benchmark-results containing results_TRANSCRIPT*/")
    parser.add_argument("--only", nargs="*", help="Subset of figures, e.g. fig1 fig3")
    args = parser.parse_args()
    os.chdir(REPO)
    setup_fonts()
    trace = TraceRecorder("make_figures", REPO / "artifacts/traces")
    trace.emit("started", published=str(args.published), our_dirs=OUR_RESULT_DIRS)
    jobs = {"fig1": lambda: fig1(args.published, trace), "fig2": lambda: fig2(trace),
            "fig3": lambda: fig3(trace), "fig4": lambda: fig4(trace),
            "fig5": lambda: fig5(trace), "fig6": lambda: fig6(trace)}
    summary = {}
    try:
        for name, job in jobs.items():
            if args.only and name not in args.only:
                continue
            summary[name] = job()
    except Exception as exc:
        trace.emit("failed", error=repr(exc))
        raise
    trace.emit("finished", summary=summary)
    print(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    print(f"trace: {trace.path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
