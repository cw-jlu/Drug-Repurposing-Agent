"""Streamlit demo for the drug-repurposing Agent (course presentation).

Run from the repository root:  streamlit run app/demo.py

Importing this module does not start the UI; Streamlit executes it as
``__main__``.  All data loading lives in ``demo_data.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import demo_data as dd  # noqa: E402

KIND_ICON = {"ok": "✅", "warn": "⚠️", "error": "❌", "info": "🔹"}
MODE_LABELS = {"benchmark_strict": "benchmark_strict（严格基准模式）",
               "research_open": "research_open（开放研究模式）"}
PLANNER_LABELS = {"rule": "rule（确定性规则规划器，离线）",
                  "deepseek": "deepseek（DeepSeek 函数调用规划器）"}


def _rel(path: str | None) -> str:
    if not path:
        return "—"
    try:
        return str(Path(path).resolve().relative_to(dd.ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- tab 1

def render_run(st, report: dict) -> None:
    status = report.get("status", "unknown")
    label, kind, explanation = dd.STATUS_LABELS.get(status, (status, "info", ""))
    cols = st.columns(4)
    cols[0].metric("状态", label)
    cols[1].metric("模式", report.get("mode", "—"))
    cols[2].metric("规划器", str(report.get("planner", "—")).split(":")[0])
    cols[3].metric("run_id", str(report.get("run_id", "—"))[:8])
    notify = {"ok": st.success, "warn": st.warning, "error": st.error}.get(kind, st.info)
    notify(f"**停止状态：{status}** — {explanation}")
    if report.get("missing_inputs"):
        st.write("缺少的输入：", ", ".join(report["missing_inputs"]))
    if report.get("error"):
        st.code(json.dumps(report["error"], ensure_ascii=False, indent=2), language="json")

    st.markdown(f"**请求：** {report.get('question', '')}")
    plan = report.get("plan")
    st.subheader("① 规划结果（Plan）")
    if plan:
        st.markdown(f"- **任务类型**：`{plan.get('task')}`\n- **理由**：{plan.get('rationale', '')}")
        st.dataframe([{"序号": i, "工具": c.get("name"),
                       "参数": json.dumps(c.get("arguments", {}), ensure_ascii=False)}
                      for i, c in enumerate(plan.get("calls", []), start=1)],
                     hide_index=True, width="stretch")
    else:
        st.info("没有通过校验的计划（规划被拦截或中断）。")
    if report.get("planner_metadata"):
        meta = report["planner_metadata"]
        st.caption(f"规划器元数据：provider={meta.get('provider')}，model={meta.get('model')}，"
                   f"latency={meta.get('latency_ms')} ms，tokens={meta.get('usage', {}).get('total_tokens')}")

    st.subheader("② 工具状态转移时间线")
    events, integrity, trace_path = dd.load_trace_events(report)
    for row in dd.timeline_rows(events):
        tool = f" · `{row['tool']}`" if row["tool"] else ""
        time = str(row["time"]).replace("T", " ")[:23]
        st.markdown(f"{KIND_ICON[row['kind']]} **步骤 {row['step']}：{row['label']}**{tool}  "
                    f"<span style='color:gray'>`{row['stage']}` · {time}</span>",
                    unsafe_allow_html=True)
        if row["details"]:
            with st.expander("事件详情", expanded=False):
                st.json(row["details"], expanded=True)

    st.subheader("③ 工具结果与输出文件")
    for result in report.get("tool_results", []):
        st.json(result, expanded=False)
    refs = dd.output_references(report)
    if refs:
        st.dataframe([{**r, "路径": _rel(r["路径"])} for r in refs],
                     hide_index=True, width="stretch")

    st.subheader("④ Trace")
    integrity_text = {
        "sha256_chain_v1": "JSONL trace 存在，SHA-256 哈希链校验通过",
        "legacy_unsealed": "JSONL trace 存在，但为早期未加哈希链的版本",
        "chain_mismatch_embedded_used": "JSONL 哈希链校验失败，改用报告内嵌 trace",
        "embedded_only": "未找到 JSONL trace 文件，仅显示 agent_run.json 内嵌的 trace（历史运行）",
    }.get(integrity, integrity)
    st.markdown(f"- Agent trace：`{_rel(trace_path or report.get('trace_file'))}`\n"
                f"- 完整性：{integrity_text}（{len(events)} 个事件）\n"
                f"- Demo trace：`{_rel(report.get('_demo_trace'))}`")
    with st.expander("原始 agent_run.json"):
        st.json({k: v for k, v in report.items() if not k.startswith("_")}, expanded=False)


def tab_agent(st) -> None:
    st.markdown("自然语言请求 → 规划器生成工具调用 → 白名单与参数校验 → 本地确定性工具执行 → "
                "每一步写入 trace。规划器看不到基准标签、原始矩阵或文件路径。")
    source = st.radio("运行方式", ["实时运行", "回放已保存运行"], horizontal=True)
    if source == "回放已保存运行":
        runs = dd.find_saved_runs()
        if not runs:
            st.warning("artifacts/ 与 benchmark/results/ 下没有找到 agent_run.json。")
            return
        choice = st.selectbox("选择已保存的运行", runs, format_func=_rel)
        render_run(st, dd.load_agent_run(choice))
        return

    example = st.selectbox("示例请求（可在下方修改）", dd.EXAMPLE_REQUESTS)
    question = st.text_area("自然语言请求", value=example, key=f"q_{example}", height=80)
    left, right = st.columns(2)
    mode = left.selectbox("模式", list(MODE_LABELS), format_func=MODE_LABELS.get)
    planners = ["rule"] + (["deepseek"] if dd.deepseek_available() else [])
    planner = right.radio("规划器", planners, format_func=PLANNER_LABELS.get)
    if not dd.deepseek_available():
        right.caption("未检测到环境变量 DEEPSEEK_API_KEY，DeepSeek 规划器不可用。")
    available = dd.available_inputs()
    st.markdown("**提供给 Agent 的输入**（由执行器控制，规划器只看到输入名称）")
    c1, c2 = st.columns(2)
    use_transcript = c1.checkbox("TRANSCRIPT 表达矩阵（items / users）", value=False,
                                 disabled=not available["transcript"],
                                 help="data/raw/TRANSCRIPT_dataset_v2.0.0；排序约数秒")
    use_luad = c2.checkbox("LUAD 冻结筛选输入", value=False, disabled=not available["luad"],
                           help="artifacts/reports/luad_eh3226 + manifests")
    if not available["transcript"]:
        c1.caption("本地缺少 TRANSCRIPT 数据。")
    if not available["luad"]:
        c2.caption("本地缺少 artifacts/reports/luad_eh3226（需先运行 rank_luad_eh3226.py），"
                   "LUAD 请求会安全停止为人工审核。")
    if st.button("运行 Agent", type="primary"):
        with st.spinner("Agent 运行中……"):
            try:
                st.session_state["last_run"] = dd.run_demo_agent(
                    question, mode, planner, use_transcript=use_transcript, use_luad=use_luad)
            except Exception as exc:  # shown to presenter; details are in the demo trace
                st.error(f"运行失败：{type(exc).__name__}: {exc}")
    if st.session_state.get("last_run"):
        st.divider()
        render_run(st, st.session_state["last_run"])


# --------------------------------------------------------------------------- tab 2

def tab_luad(st) -> None:
    frame, config = dd.load_luad_candidates()
    st.warning("研究假设清单，不是治疗推荐。全部 10 个候选的证据等级均为 "
               "`insufficient_evidence`；9/10 为糖皮质激素类，属于同一类别信号。")
    st.caption(f"筛选：GSE32863（57 对 LUAD 肿瘤/正常）疾病签名 × EH3226 A549（10 µM, 24 h）药物扰动；"
               f"负 Spearman + 上/下调基因集连通性，RRF k=60。证据复核日期：{config.get('reviewed_at')}")
    if not config["scores_available"]:
        st.info("本地没有 artifacts/reports/luad_eh3226/all_candidates.csv，分数列显示为空；"
                "运行 scripts/rank_luad_eh3226.py 后会自动显示 Spearman/连通性/RRF 分数。")
    lines = ["| 排名 | 药物 | RRF | 靶点 | 通路 | 证据等级 | 候选 PMID |",
             "| ---: | --- | ---: | --- | --- | --- | --- |"]
    for row in frame.itertuples():
        rrf = "—" if row.rrf != row.rrf else f"{row.rrf:.5f}"
        lines.append(f"| {row.rank} | {row.name} | {rrf} | {', '.join(row.targets) or '未知'} | "
                     f"{', '.join(row.pathways) or '未知'} | `{row.confidence}` | "
                     f"{dd.pmid_links_md(row.candidate_pmids)} |")
    st.markdown("\n".join(lines))
    st.markdown(f"**共享机制：** {config.get('shared_mechanism')}  \n"
                f"**共享支持性 PMID：** {dd.pmid_links_md(config.get('shared_supporting_pmids', []))}  \n"
                f"**共享反对性 PMID：** {dd.pmid_links_md(config.get('shared_contradicting_pmids', []))}")

    st.subheader("候选证据卡片")
    for row in frame.itertuples():
        with st.expander(f"#{row.rank} {row.name} — {row.confidence}"):
            if row.rrf == row.rrf:
                a, b, c = st.columns(3)
                a.metric("Spearman 反转", f"{row.spearman_reversal:.4f}")
                b.metric("连通性", f"{row.connectivity:.4f}")
                c.metric("RRF", f"{row.rrf:.5f}")
            st.markdown(
                f"- **来源签名**：`{getattr(row, 'sig_id', '—')}`（pert_id `{getattr(row, 'pert_id', '—')}`）\n"
                f"- **身份审计**：`{row.identity}`\n"
                f"- **身份状态**：{row.identity_status}\n"
                f"- **靶点 / 机制**：{row.targets_mechanism}\n"
                f"- **LUAD/NSCLC 证据**：{row.evidence_found}\n"
                f"- **反对证据 / 不确定性**：{row.evidence_against}\n"
                f"- **结论**：{row.decision}\n"
                f"- **候选特异 PMID**：{dd.pmid_links_md(row.candidate_pmids)}")
    with st.expander("证据矩阵局限性（configs/luad_top10_evidence_v1.json）"):
        st.markdown("\n".join(f"- {x}" for x in config.get("limitations", [])))


# --------------------------------------------------------------------------- tab 3

def tab_benchmark(st) -> None:
    import altair as alt

    summary, per_seed, notes = dd.load_benchmark()
    st.markdown("RECeSS 官方 runner，TRANSCRIPT v2.0.0，N=100 个种子，5 折，ptest=0.2；"
                "主指标为作者论文的 NS-AUC（`Lin's AUC`）。B2 为本项目方法，其余 11 个为作者发布的模型结果。")
    for note in notes:
        st.caption(note)
    metrics = sorted(set(summary.metric), key=lambda m: (m != "NS-AUC", m))
    metric = st.selectbox("指标", metrics)
    table = dd.ranking_table(summary, metric)
    ours = {"B2"} | set(summary[summary.source != "official_json"].model)
    st.dataframe(table.style.format(precision=4).apply(
        lambda r: ["font-weight: bold; background-color: rgba(255,200,0,0.25)"
                   if r["模型"] in ours else "" for _ in r], axis=1),
        hide_index=True, width="stretch")

    split = st.radio("划分", dd.SPLITS, format_func=dd.SPLIT_LABELS.get, horizontal=True)
    part = summary[(summary.metric == metric) & (summary.split == split)].copy()
    part["low"], part["high"] = part["mean"] - part["sd"], part["mean"] + part["sd"]
    part["组别"] = part.model.map(lambda m: "本项目" if m in ours else "已发表模型")
    order = part.sort_values("mean", ascending=False).model.tolist()
    base = alt.Chart(part).encode(y=alt.Y("model:N", sort=order, title=None))
    bars = base.mark_bar().encode(
        x=alt.X("mean:Q", title=f"{metric} 均值 ± SD"),
        color=alt.Color("组别:N", scale=alt.Scale(domain=["本项目", "已发表模型"],
                                                  range=["#e4572e", "#8da0cb"])),
        tooltip=["model", alt.Tooltip("mean:Q", format=".4f"), alt.Tooltip("sd:Q", format=".4f")])
    errors = base.mark_errorbar().encode(x="low:Q", x2="high:Q")
    rule = alt.Chart().mark_rule(strokeDash=[4, 4], color="gray").encode(x=alt.datum(0.5))
    st.altair_chart(alt.layer(bars, errors, rule, data=part).properties(height=380),
                    width="stretch")
    st.caption("虚线为 0.5（随机水平）。weakly correlated 划分在 100 个种子中重复同一个外部测试集。")

    seeds = per_seed[(per_seed.metric == metric) & (per_seed.split == split)].dropna(subset=["value"])
    if not seeds.empty:
        st.markdown("**逐种子分布（本地有原始 CSV 的模型）**")
        box = alt.Chart(seeds).mark_boxplot(extent="min-max").encode(
            x=alt.X("model:N", title=None), y=alt.Y("value:Q", title=metric,
                                                   scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", legend=None))
        st.altair_chart(box.properties(height=280), width="stretch")


# --------------------------------------------------------------------------- tab 4

KEY_LIMITATIONS = [
    "**反转分数 ≠ 疗效**：表达谱反向匹配只衡量表达相反程度，不代表患者获益、剂量可行性或安全性。",
    "**基准表现有限**：官方 100 次运行中 B2 的 NS-AUC 为 0.5222（random simple，9/12）与 0.5019"
    "（weakly correlated，8/12），与最强模型 BNNR / MBiRW 相差约 0.21–0.24，100 个配对种子中 0 胜。",
    "**weakly correlated 划分**在 100 个种子中重复同一个外部测试集，区间反映内部折/模型选择波动，而非 100 次独立划分。",
    "**LUAD Top-10 全部为 insufficient_evidence**：9/10 为糖皮质激素类同一类别信号；仅 2 个与 Broad Hub InChIKey 完全一致；"
    "没有候选特异的 LUAD 动物或临床疗效证据，也没有湿实验数据。",
    "**文献检索有边界**：“未找到研究”只表示在限定检索中未找到，不代表不存在。",
    "**LLM 结论范围有限**：路由评测（v3 holdout：DeepSeek 98/100，规则 90/100）只衡量工具选择；"
    "两次模型选择不能证明一般化能力；五分区方法选择实验为负结果（0.40653 vs 固定 B2 0.51684）。",
    "**安全边界**：处方、剂量、患者建议等请求一律转人工审核；Agent 不给出治疗推荐。",
]


def tab_limitations(st) -> None:
    st.markdown("\n".join(f"{i}. {text}" for i, text in enumerate(KEY_LIMITATIONS, start=1)))
    with st.expander("docs/limitations.md 原文"):
        st.markdown("\n".join(f"- {x}" for x in dd.load_limitations()))


def main() -> None:
    import streamlit as st

    os.chdir(dd.ROOT)  # package code resolves data/ and artifacts/ relative to the repo root
    st.set_page_config(page_title="药物重定位 Agent 演示", page_icon="💊", layout="wide")
    st.title("💊 可审计的药物重定位 Agent")
    st.caption("转录组反向匹配 · 白名单工具调用 · 全程 trace · 仅供研究，不构成治疗建议")
    tabs = st.tabs(["Agent 运行", "LUAD 候选", "Benchmark", "局限性"])
    with tabs[0]:
        tab_agent(st)
    with tabs[1]:
        tab_luad(st)
    with tabs[2]:
        tab_benchmark(st)
    with tabs[3]:
        tab_limitations(st)


if __name__ == "__main__":
    main()
