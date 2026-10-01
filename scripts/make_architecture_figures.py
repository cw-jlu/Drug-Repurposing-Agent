"""Architecture diagram (fig9) and LUAD workflow flowchart (fig10).

Component, tool and status names follow src/drug_repurposing_agent/agent.py
(TOOLS: rank_transcriptome, package_luad_case, manual_review; statuses
planning/blocked/needs_input/manual_review_required/completed/failed) and the
committed scripts. Numbers come from the committed result files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
NAVY, TEAL, RED, INK, MUTED = "#112B3C", "#087E78", "#9E493D", "#183042", "#5B6B75"
FILL = {"llm": "#E3F1EF", "code": "#E8EDF2", "tool": "#F4EFE6", "out": "#F2E7E4", "user": NAVY}
OUT = Path("docs/figures")


def box(ax, x, y, w, h, title, body="", kind="code", title_color=None, size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.016",
                                fc=FILL[kind], ec=NAVY if kind == "user" else "#9AA8B1", lw=1.0))
    color = "white" if kind == "user" else (title_color or NAVY)
    if body:
        ax.text(x + w / 2, y + h - 0.02, title, ha="center", va="top", fontsize=size, weight="bold", color=color)
        ax.text(x + w / 2, y + h - 0.052, body, ha="center", va="top", fontsize=8.6,
                color="white" if kind == "user" else INK, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size, weight="bold", color=color)


def arrow(ax, a, b, color=MUTED, lw=1.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw))


def path(ax, pts, color=MUTED, lw=1.2):
    """Orthogonal polyline ending in an arrow head."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="butt")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def label(ax, x, y, text, color=MUTED, size=8.6, **kw):
    ax.text(x, y, text, fontsize=size, color=color, **kw)


def legend(ax, x, y):
    for i, (k, t) in enumerate((("llm", "LLM 参与"), ("code", "普通代码"), ("tool", "确定性工具"), ("out", "输出/停止"))):
        ax.add_patch(FancyBboxPatch((x + i * 0.11, y), 0.018, 0.016, boxstyle="round,pad=0.002",
                                    fc=FILL[k], ec="#9AA8B1"))
        label(ax, x + 0.024 + i * 0.11, y + 0.001, t, color=INK, size=9)


def architecture() -> None:
    planner = json.loads(Path("benchmark/results/planner_eval_v3_deepseek_flash.json").read_text(encoding="utf-8"))
    p = next(r for r in planner["results"] if r["planner"].startswith("deepseek"))
    fig, ax = plt.subplots(figsize=(15, 9.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.995, "系统架构：LLM 只负责理解与编排，数值计算交给确定性工具，代码做最终校验", ha="center",
            va="top", fontsize=14, weight="bold", color=NAVY)
    legend(ax, 0.29, 0.935)
    for y, name in ((0.81, "交互层"), (0.625, "编排层"), (0.43, "工具层"), (0.235, "校验层"), (0.075, "输出层")):
        label(ax, 0.006, y, name, color=MUTED, size=10, weight="bold", rotation=90, va="center")

    # Row 1
    y1, h1 = 0.735, 0.16
    box(ax, 0.04, y1 + 0.015, 0.16, 0.13, "自然语言请求", "例：请为肺腺癌筛选候选药物\n模式：Strict / Open", kind="user")
    box(ax, 0.26, y1, 0.22, h1, "Planner（LLM）",
        f"DeepSeek 函数调用；规则规划器兜底\n只看到：模式 + 可用输入名称\n看不到：数据、文件路径、标签\n冻结 holdout {p['correct']}/100",
        kind="llm", title_color=TEAL)
    box(ax, 0.54, y1, 0.2, h1, "工具调用校验（代码）",
        "工具名必须在白名单内\n参数符合 JSON Schema\n当前模式是否允许该工具\n缺输入 → needs_input", kind="code")
    box(ax, 0.8, y1, 0.17, h1, "决策门控",
        "规则算结构化字段\nLLM 只判读文字备注\n置信度不足 → 转人工\n（Jev 接口已预留）", kind="llm", title_color=TEAL)
    arrow(ax, (0.2, y1 + h1 / 2), (0.26, y1 + h1 / 2))
    arrow(ax, (0.48, y1 + h1 / 2), (0.54, y1 + h1 / 2))
    arrow(ax, (0.74, y1 + h1 / 2), (0.8, y1 + h1 / 2))

    # Row 2: state machine band
    yb, hb = 0.58, 0.09
    ax.add_patch(FancyBboxPatch((0.04, yb), 0.93, hb, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc="#FFFFFF", ec="#C9D2D8", lw=0.9, ls="--"))
    label(ax, 0.05, yb + 0.058, "状态机（每一步写入 SHA-256 链式 JSONL 轨迹）", color=NAVY, size=9.5, weight="bold")
    states = [("planning", MUTED), ("completed", TEAL), ("needs_input", RED), ("blocked", RED),
              ("manual_review_required", RED), ("failed", RED)]
    for (s, c), x in zip(states, (0.06, 0.19, 0.33, 0.48, 0.61, 0.86)):
        ax.text(x, yb + 0.018, s, fontsize=9, color=c, weight="bold", family="monospace")
    arrow(ax, (0.64, y1), (0.64, yb + hb))
    arrow(ax, (0.885, y1), (0.885, yb + hb))

    # Row 3: tools
    y3, h3 = 0.33, 0.2
    tools = [(0.04, 0.28, "rank_transcriptome（Strict + Open）",
              "按基因 ID 对齐药物/疾病表达矩阵\n负 Spearman 反转 + 上下调基因集连接性\nRRF 融合（k=60），不读任何标签\nBenchmark 适配：B2 / B3 / B4（含 BNNR 移植）", "tool", RED),
             (0.36, 0.28, "package_luad_case（仅 Open）",
              "核对冻结输入/输出哈希\n打包 Top-10、身份审计、阳性对照\n证据账本与执行轨迹\nmanual_review：不支持的任务安全停止", "tool", RED),
             (0.68, 0.29, "证据层（多 Agent）",
              "文献 Agent：PubMed 检索 + 原文引语\n批评 Agent：独立检索反对证据\n协调者：给出证据分级", "llm", TEAL)]
    for x, w, t, b, k, c in tools:
        box(ax, x, y3, w, h3, t, b, kind=k, title_color=c)
        arrow(ax, (x + w / 2, yb), (x + w / 2, y3 + h3))
    label(ax, 0.512, yb - 0.03, "completed 路径按计划调用工具", color=MUTED, size=8.6, ha="left")

    # Row 4: validator spanning full width
    y4, h4 = 0.19, 0.09
    box(ax, 0.04, y4, 0.93, h4, "确定性 Validator（代码）",
        "Schema、ID、哈希、分数与排名一致性 · PMID 必须在检索集中、引语逐字出现 · 工具权限与预算", kind="code")
    for x, w, *_ in tools:
        arrow(ax, (x + w / 2, y3), (x + w / 2, y4 + h4))

    # Row 5: outputs
    y5, h5 = 0.025, 0.11
    box(ax, 0.04, y5, 0.44, h5, "研究输出",
        "613×151 分数矩阵 · LUAD Top-10 证据卡 · agent_run.json + 链式轨迹", kind="out", title_color=RED)
    box(ax, 0.53, y5, 0.44, h5, "外部评测（隔离）",
        "RECeSS 官方 Runner 只读取分数矩阵 · 标签仅在评分器内使用", kind="out", title_color=RED)
    arrow(ax, (0.26, y4), (0.26, y5 + h5))
    arrow(ax, (0.75, y4), (0.75, y5 + h5))
    fig.savefig(OUT / "fig9_architecture.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig9_architecture.svg", bbox_inches="tight")
    plt.close(fig)


def diamond(ax, cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, fc="#FFFFFF", ec=TEAL, lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.6, color=INK, linespacing=1.25)


def workflow() -> None:
    enr = json.loads(Path("benchmark/results/luad_pathway_enrichment_v1.json").read_text(encoding="utf-8"))
    rev = json.loads(Path("benchmark/results/multi_agent_review_v1.json").read_text(encoding="utf-8"))
    pc = json.loads(Path("benchmark/results/luad_positive_control_stats_v1.json").read_text(encoding="utf-8"))
    up, down = enr["signature_sizes"]["up"], enr["signature_sizes"]["down"]
    fig, ax = plt.subplots(figsize=(15, 10.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.995, "端到端流程：从自然语言请求到肺腺癌候选药物证据报告", ha="center", va="top",
            fontsize=14, weight="bold", color=NAVY)
    legend(ax, 0.29, 0.948)
    W, H, DW, DH = 0.16, 0.095, 0.15, 0.105
    col = [0.02, 0.22, 0.42, 0.62, 0.82]
    cx = [c + W / 2 for c in col]
    T = lambda t, b, k, c=None: (t, b, k, c)

    # Row 1: request -> plan -> check -> inputs -> (stop)
    y1 = 0.80; m1 = y1 + H / 2
    box(ax, col[0], y1, W, H, "1. 自然语言请求", "“请为肺腺癌筛选候选药物”", kind="user", size=10)
    box(ax, col[1], y1, W, H, "2. Planner 生成计划", "LLM 选择允许的工具\n与参数", kind="llm", title_color=TEAL, size=10)
    diamond(ax, cx[2], m1, DW, DH, "3. 工具与参数\n通过校验？")
    diamond(ax, cx[3], m1, DW, DH, "4. 所需输入\n都已提供？")
    box(ax, col[4], y1, W, H, "停止并记录", "blocked / needs_input\n写明原因", kind="out", title_color=RED, size=10)
    arrow(ax, (col[0] + W, m1), (col[1], m1))
    arrow(ax, (col[1] + W, m1), (cx[2] - DW / 2, m1))
    arrow(ax, (cx[2] + DW / 2, m1), (cx[3] - DW / 2, m1)); label(ax, cx[2] + DW / 2 + 0.008, m1 + 0.008, "是", color=TEAL)
    arrow(ax, (cx[3] + DW / 2, m1), (col[4], m1), color=RED); label(ax, cx[3] + DW / 2 + 0.008, m1 + 0.008, "否", color=RED)
    # 3 "no": up and over to the stop box
    path(ax, [(cx[2], m1 + DH / 2), (cx[2], 0.925), (cx[4], 0.925), (cx[4], y1 + H)], color=RED)
    label(ax, cx[2] + 0.006, 0.905, "否", color=RED)
    # 4 "yes": down then left to step 5
    y2 = 0.585
    lane12 = 0.735
    path(ax, [(cx[3], m1 - DH / 2), (cx[3], lane12), (cx[0], lane12), (cx[0], y2 + H)], color=TEAL)
    label(ax, cx[3] + 0.006, lane12 + 0.012, "是：Open 模式执行 LUAD 流程", color=TEAL)

    # Row 2: data -> DEG -> pathways -> drug signatures -> scoring
    row2 = [T("5. 疾病数据质控", "GSE32863：核实 57 对\n肿瘤/正常，排除 2 份", "tool", RED),
            T("6. 配对差异表达", f"配对检验 + BH 校正\n上调 {up} / 下调 {down}", "tool", RED),
            T("7. 通路富集", "Hallmark/KEGG\n增殖、糖酵解↑；炎症↓", "tool", RED),
            T("8. 药物扰动签名", "LINCS A549，10 µM，24 h\n4,920 个药 × 961 基因", "tool", RED),
            T("9. 反转打分与融合", "负 Spearman + 基因集连接性\nRRF → 全部药物排名", "tool", RED)]
    for c, (t, b, k, tc) in zip(col, row2):
        box(ax, c, y2, W, H, t, b, kind=k, title_color=tc, size=10)
    for a, b in zip(col[:-1], col[1:]):
        arrow(ax, (a + W, y2 + H / 2), (b, y2 + H / 2))

    # Row 3 (right to left): Top-10 -> audit -> literature -> critic -> citation check
    y3 = 0.37; m3 = y3 + H / 2
    row3 = [(col[4], T("10. Top-10 候选", "排名冻结；阈值敏感性\n作为稳健性披露", "tool", RED)),
            (col[3], T("11. 身份与对照审计", f"签名 ID 溯源 / Hub InChIKey\n参考药恢复 p = {pc['permutation_p_one_sided']:.2f}", "code")),
            (col[2], T("12. 文献 Agent", "PubMed 检索\n提出带原文引语的论断", "llm", TEAL)),
            (col[1], T("13. 批评 Agent", "独立检索反对证据\n逐条质疑支持论断", "llm", TEAL))]
    for c, (t, b, k, tc) in row3:
        box(ax, c, y3, W, H, t, b, kind=k, title_color=tc, size=10)
    diamond(ax, cx[0], m3, DW, DH + 0.01, "14. PMID 在检索集中\n且引语逐字出现？")
    arrow(ax, (cx[4], y2), (cx[4], y3 + H))
    for a, b in ((col[4], col[3]), (col[3], col[2]), (col[2], col[1])):
        arrow(ax, (a, m3), (b + W, m3))
    arrow(ax, (col[1], m3), (cx[0] + DW / 2, m3))
    label(ax, cx[0] + 0.012, y3 - 0.032, f"否 → 丢弃该论断（{rev['citation_validation']['proposed_quoted_items']} 条中 "
          f"{rev['citation_validation']['rejected_quoted_items']} 条被拒）", color=RED)

    # Row 4: grade -> gate -> validator -> report; human review below the gate
    y4 = 0.17; m4 = y4 + H / 2
    box(ax, col[0], y4, W, H, "15. 协调者分级", "SUPPORTED / PROMISING /\nCONFLICTING / INSUFFICIENT", kind="llm", title_color=TEAL, size=10)
    diamond(ax, cx[1], m4, DW, DH + 0.01, "16. 置信度够高\n且非高风险？")
    box(ax, col[2], y4, W, H, "17. 确定性 Validator", "哈希、ID、排名、引用、权限\n全部通过才打包", kind="code", size=10)
    box(ax, col[3], y4, W + 0.2, H, "18. 证据报告",
        f"Top-10 证据卡：{rev['tier_counts']['INSUFFICIENT_EVIDENCE']}/10 证据不足 · 研究假设，非用药建议", kind="out", title_color=RED, size=10)
    box(ax, col[1], 0.02, W, 0.085, "转人工复核", "manual_review_required", kind="out", title_color=RED, size=10)
    path(ax, [(cx[0], m3 - (DH + 0.01) / 2), (cx[0], y4 + H)], color=TEAL)
    label(ax, cx[0] - 0.022, y3 - 0.06, "是", color=TEAL)
    arrow(ax, (col[0] + W, m4), (cx[1] - DW / 2, m4))
    arrow(ax, (cx[1] + DW / 2, m4), (col[2], m4), color=TEAL); label(ax, cx[1] + DW / 2 + 0.006, m4 + 0.008, "是", color=TEAL)
    arrow(ax, (cx[1], m4 - (DH + 0.01) / 2), (cx[1], 0.105), color=RED); label(ax, cx[1] + 0.008, 0.125, "否", color=RED)
    arrow(ax, (col[2] + W, m4), (col[3], m4))

    ax.text(0.71, 0.06, "全程：每一步写入 SHA-256 链式轨迹；LLM 不接触数值计算\nBenchmark（Strict）模式只允许 rank_transcriptome，"
            "LLM 看不到药名与标签", ha="center", va="center", fontsize=9.3, color=NAVY, linespacing=1.5,
            bbox={"boxstyle": "round,pad=0.5", "fc": "#F3F6F8", "ec": "#C9D2D8"})
    fig.savefig(OUT / "fig10_workflow.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig10_workflow.svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    workflow()
    print("wrote fig9_architecture and fig10_workflow")
