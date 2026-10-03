"""Streamlit demo for the drug-repurposing Agent (course presentation).

Run from the repository root:  streamlit run app/demo.py

Importing this module does not start the UI; Streamlit executes it as
``__main__``.  All data loading lives in ``demo_data.py``.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
# The repository root must be importable too: agent v2's real tools import from
# the repo's scripts/ and evals/ packages (Streamlit only adds app/ to sys.path).
for _path in (APP_DIR.parent, APP_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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


# --------------------------------------------------------------------------- style

# Palette shared with the deck and the paper figures (scripts/figure_style.py).
CSS = """
<style>
:root { --navy:#112B3C; --ink:#183042; --muted:#5B6B75; --teal:#087E78; --red:#9E493D;
        --amber:#B9862E; --line:#DDE3E6; --sand:#EEF3F2; }
.block-container { padding-top: 1.6rem; max-width: 1240px; }
h1, h2, h3 { color: var(--navy); letter-spacing: 0; }
.hero-sub { color: var(--muted); margin: -0.4rem 0 1rem 0; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin: 0 0 1.2rem 0; }
.kpi { border:1px solid var(--line); border-radius:12px; padding:12px 16px; background:#fff; }
.kpi .v { font-size:1.7rem; font-weight:700; color:var(--navy); line-height:1.2; }
.kpi .l { color:var(--ink); font-size:.9rem; font-weight:600; }
.kpi .n { color:var(--muted); font-size:.78rem; }
.banner { border-radius:12px; padding:12px 16px; margin:6px 0 14px 0; border-left:6px solid; color:var(--ink); }
.banner.ok { background:#E3F1EF; border-color:var(--teal); }
.banner.warn { background:#F6EEDC; border-color:var(--amber); }
.banner.error { background:#F2E7E4; border-color:var(--red); }
.banner b { color:var(--navy); }
.round { margin: 4px 0 12px 0; }
.round .head { font-size:.9rem; color:var(--muted); margin-bottom:6px; }
.flow { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.chip { padding:5px 11px; border-radius:999px; font-size:.85rem; border:1px solid; white-space:nowrap; }
.chip small { opacity:.75; margin-left:4px; }
.chip.ok { background:#E3F1EF; border-color:var(--teal); color:#065F5A; }
.chip.skipped_already_done { background:#F4F6F7; border-color:#B9C2C7; color:var(--muted); }
.chip.failed { background:#F2E7E4; border-color:var(--red); color:#7A3329; }
.chip.stopped { background:#F6EEDC; border-color:var(--amber); color:#7A5A1E; }
.chip.not_run { background:#fff; border-color:#C9D2D8; color:#8A979F; border-style:dashed; }
.arrow { color:#B9C2C7; }
.tag { display:inline-block; padding:2px 9px; margin:2px 4px 2px 0; border-radius:6px; font-size:.82rem;
       background:var(--sand); color:var(--ink); }
.tag.up { background:#F2E7E4; color:#7A3329; } .tag.down { background:#E2ECF4; color:#22506F; }
.tag.ok { background:#E3F1EF; color:#065F5A; } .tag.info { background:#E2ECF4; color:#22506F; }
.tag.warn { background:#F6EEDC; color:#7A5A1E; } .tag.muted { background:#F4F6F7; color:var(--muted); }
.card { border:1px solid var(--line); border-radius:12px; padding:14px 16px; background:#fff; height:100%; }
.card h4 { margin:0 0 4px 0; color:var(--navy); font-size:1.05rem; }
.card .meta { color:var(--muted); font-size:.85rem; margin-bottom:6px; }
.card .note { color:var(--muted); font-size:.8rem; margin-top:8px; }
</style>
"""


def _e(text: object) -> str:
    """Escape report text before it goes into raw HTML."""
    return html.escape(str(text), quote=True)


def kpi_cards(st) -> None:
    cells = "".join(f'<div class="kpi"><div class="v">{_e(v)}</div><div class="l">{_e(l)}</div>'
                    f'<div class="n">{_e(n)}</div></div>' for v, l, n in dd.headline_kpis())
    st.markdown(f'<div class="kpis">{cells}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- tab: agent v2

def _banner(st, report: dict) -> None:
    status = report.get("status", "planning")
    label, kind = dd.V2_STATUS.get(status, (status, "error"))
    disease = dd.v2_disease(report)
    stop = [e.get("reason") for e in report.get("executed", []) if e.get("tool") == "manual_review"]
    detail = stop[0] if stop and stop[0] else report.get("stop_reason", "")
    rounds = len(dd.v2_pipeline(report))
    planner = str(report.get("planner", "—")).split(":")[0]
    if not disease and status != "completed":
        names = "、".join(c["label"] for c in dd.registry_cards())
        detail = f"请求中没有识别到登记疾病（已登记：{names}），疾病分析工具不可用。" + (
            f"规划器说明：{detail}" if detail else "")
    st.markdown(
        f'<div class="banner {kind}"><b>{_e(label)}</b>　·　疾病：{_e(disease or "未识别到登记疾病")}'
        f'　·　规划器：{_e(planner)}　·　规划 {rounds} 轮'
        + (f'<br><span style="font-size:.88rem">{_e(detail)}</span>' if detail else "")
        + f'<br><span style="font-size:.85rem;color:#5B6B75">请求：{_e(report.get("question", ""))}</span></div>',
        unsafe_allow_html=True)


def _pipeline(st, report: dict) -> None:
    for r in dd.v2_pipeline(report):
        if r["invalid_reason"]:
            head = f"第 {r['round']} 轮 · 计划被代码校验拒绝：{_e(r['invalid_reason'])}"
        elif r["error"]:
            head = f"第 {r['round']} 轮 · 规划器调用失败：{_e(r['error'])}"
        else:
            head = f"第 {r['round']} 轮 · 校验通过"
        chips = '<span class="arrow">→</span>'.join(
            f'<span class="chip {s["status"]}">{_e(s["label"])}<small>{_e(dd.V2_STEP_STATUS.get(s["status"], s["status"]))}</small></span>'
            for s in r["steps"]) or '<span class="chip not_run">（空计划）</span>'
        st.markdown(f'<div class="round"><div class="head">{head}</div><div class="flow">{chips}</div></div>',
                    unsafe_allow_html=True)


def _tags(items, kind: str = "") -> str:
    return "".join(f'<span class="tag {kind}">{_e(x)}</span>' for x in items)


def _outputs(st, report: dict) -> None:
    out = dd.v2_outputs(report)
    if not any(out[k] for k in ("cohort", "signature", "ranking", "benchmark")):
        st.info("这次运行没有产生分析结果（未执行任何分析工具）。")
        return
    cols = st.columns(4)
    coh, sig, rk, au = out["cohort"] or {}, out["signature"] or {}, out["ranking"] or {}, out["audit"] or {}
    if coh:
        n = f"{coh['pairs']} 对" if "pairs" in coh else f"{coh.get('tumor')} vs {coh.get('normal')}"
        cols[0].metric("纳入样本", n, help=f"共 {coh.get('samples')} 个样本，排除 {coh.get('excluded')} 个")
    if sig:
        cols[1].metric("差异基因 上调 / 下调", f"{sig.get('up')} / {sig.get('down')}",
                       help="FDR < 0.05 且 |log2FC| ≥ 1，从原始 GEO 数据重算")
    if rk:
        cols[2].metric("排名化合物", f"{rk.get('drugs'):,}", help=f"LINCS {rk.get('cell_line')} 细胞系")
    if au:
        if "permutation_p" in au:
            cols[3].metric("参考药检验 p", f"{au['permutation_p']:.3f}",
                           help=f"{au.get('measured_controls')}/{au.get('reference_drugs_listed', '?')} 个预先登记的参考药可测；"
                                f"平均名次百分位 {au.get('mean_percentile')}（0.5 = 随机）")
        else:
            cols[3].metric("参考药检验", "无法计算", help=str(au.get("reference_drugs", "")))
    if out["fetch"]:
        files = out["fetch"].get("files", {})
        st.caption("数据：" + "，".join(f"{k} {'已下载并核验哈希' if v.startswith('downloaded') else '本地已核验'}"
                                      for k, v in files.items()))
    if out["pathways"]:
        st.markdown("**Hallmark 通路**　" + _tags(out["pathways"].get("up", []), "up")
                    + _tags(out["pathways"].get("down", []), "down"), unsafe_allow_html=True)
        st.caption("红色：肿瘤中上调基因富集的通路；蓝色：下调基因富集的通路")
    if rk.get("top10"):
        tiers = out["tiers"] or {}
        measured = set(au.get("measured_names", []))
        rows = []
        for i, name in enumerate(rk["top10"], 1):
            row = {"排名": i, "化合物": name}
            if tiers:
                tier = tiers.get(name)
                row["文献分级"] = dd.V2_TIER_LABELS.get(tier, (tier or "—", ""))[0]
            if measured:
                row["预先登记的参考药"] = "是" if name in measured else ""
            rows.append(row)
        st.markdown("**Top-10 候选**（研究假设，不是用药建议）")
        st.dataframe(rows, hide_index=True, width="stretch",
                     column_config={"排名": st.column_config.NumberColumn(width="small")})
        if out["evidence_note"]:
            note = out["evidence_note"]
            note = ("该疾病没有冻结的文献审阅，且未开启实时审阅，因此没有做文献审阅（可在右侧打开“实时联网文献审阅”）"
                    if note.startswith("literature review not available") else
                    "这次请求没有做文献审阅" if note == "literature review not run" else note)
            st.caption(f"文献：{note}")
        elif out["review"] and not tiers:
            counts = out["review"].get("tier_counts", {})
            st.markdown("**文献分级汇总**　" + _tags(
                f"{dd.V2_TIER_LABELS.get(k, (k, ''))[0]} {v}" for k, v in counts.items() if v), unsafe_allow_html=True)
    if out["benchmark"]:
        st.markdown(f"**基准排名**：{_e(out['benchmark'].get('status'))}，结果清单 `{_e(out['benchmark'].get('manifest'))}`")


def render_v2(st, report: dict) -> None:
    _banner(st, report)
    st.markdown("##### 规划与执行")
    _pipeline(st, report)
    t1, t2, t3, t4 = st.tabs(["结论摘要", "规划理由", "执行记录", "原始记录"])
    with t1:
        _outputs(st, report)
    with t2:
        for r in dd.v2_pipeline(report):
            if r["rationale"]:
                st.markdown(f"**第 {r['round']} 轮**：{r['rationale']}")
        if not any(r["rationale"] for r in dd.v2_pipeline(report)):
            st.caption("该运行没有保存规划理由（规则规划器或归档摘要）。")
        st.caption("提供给规划器的输入名称：" + ", ".join(report.get("available_inputs", [])))
    with t3:
        st.dataframe(dd.v2_step_rows(report), hide_index=True, width="stretch")
    with t4:
        if report.get("_path"):
            st.caption(f"运行记录：`{_rel(report['_path'])}` · Demo trace：`{_rel(report.get('_demo_trace'))}`")
        st.json({k: v for k, v in report.items() if not k.startswith("_")}, expanded=False)


def tab_agent_v2(st) -> None:
    st.markdown("一句话描述需求，Agent 会规划步骤、经代码校验后调用真实工具执行。"
                "只分析**登记表中的 5 个疾病**；其他疾病会安全转人工。")
    source = st.segmented_control("运行方式", ["实时运行", "回放已保存运行"], default="实时运行",
                                  key="v2_source", label_visibility="collapsed")
    if source == "回放已保存运行":
        runs = dd.saved_v2_runs()
        if not runs:
            st.warning("没有找到已保存的 v2 运行。")
            return
        choice = st.selectbox("选择运行", list(runs), key="v2_replay")
        st.divider()
        render_v2(st, runs[choice])
        return
    labels = list(dd.V2_EXAMPLE_LABELS)
    example = st.pills("示例请求（点击填入）", labels, key="v2_example", default=labels[0])
    text = dd.V2_EXAMPLE_LABELS.get(example, "")
    left, right = st.columns([3, 2], gap="large")
    with left:
        question = st.text_area("自然语言请求（可修改）", value=text, key=f"v2_q_{example}", height=150)
    with right:
        st.markdown("**设置**")
        planners = (["deepseek"] if dd.deepseek_v2_available() else []) + ["rule"]
        planner = st.segmented_control("规划器", planners, default=planners[0], key="v2_planner",
                                       format_func={"deepseek": "DeepSeek（默认）", "rule": "规则（离线）"}.get)
        if "deepseek" not in planners:
            st.caption("未检测到 DeepSeek 密钥，只能使用规则规划器。")
        live = st.toggle("实时联网文献审阅", value=False, disabled="deepseek" not in planners, key="v2_live",
                         help="PubMed + DeepSeek，约 20–30 次调用、3–4 分钟；关闭时肺腺癌复用冻结审阅，其他疾病不做文献审阅")
        with st.expander("高级设置"):
            mode = st.selectbox("模式", list(MODE_LABELS), index=1, format_func=MODE_LABELS.get, key="v2_mode")
        run = st.button("运行 Agent", type="primary", width="stretch", disabled=not (question or "").strip())
    if run:
        with st.status("Agent 运行中：规划 → 校验 → 下载/质控 → 差异表达 → 排名 → 报告 …", expanded=False) as box:
            try:
                st.session_state["v2_last"] = dd.run_v2(question, planner or planners[0], mode, live_review=live)
                box.update(label="运行结束", state="complete")
            except Exception as exc:
                box.update(label=f"运行失败：{type(exc).__name__}", state="error")
                st.error(f"运行失败：{type(exc).__name__}: {exc}")
    if st.session_state.get("v2_last"):
        st.divider()
        render_v2(st, st.session_state["v2_last"])


# --------------------------------------------------------------------------- tab: registry

def tab_registry(st) -> None:
    st.markdown("每个疾病在登记时已固定 GEO 文件地址与 SHA-256、样本分组规则、药物细胞系、参考药清单和文献检索配置；"
                "数据只在 Agent 需要时下载。上限取决于药物端是否有匹配的 LINCS 细胞系。")
    cards = dd.registry_cards()
    for i in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, c in zip(cols, cards[i:i + 3]):
            disk = '<span class="tag ok">本地已有</span>' if c["on_disk"] else '<span class="tag muted">用到时下载</span>'
            col.markdown(
                f'<div class="card"><h4>{_e(c["label"])}</h4>'
                f'<div class="meta">{_e(c["accession"])} · {_e(c["platform"])} · 药物端 {_e(c["cell_line"])}</div>'
                f'<span class="tag">{_e(c["design"])}</span><span class="tag">参考药 {c["reference_drugs"]} 个</span>'
                f'<span class="tag">{_e(c["literature"])}</span>{disk}'
                f'<div class="note">别名：{_e("、".join(c["aliases"]))}<br>{_e(c["note"])}</div></div>',
                unsafe_allow_html=True)
    fig = dd.ROOT / "docs" / "figures" / "fig11_registry_overview.png"
    if fig.is_file():
        st.markdown("##### 端到端验证结果")
        st.image(str(fig), width="stretch",
                 caption="(a) 从原始 GEO 数据重算的疾病签名；(b) 预先登记参考药的平均名次百分位（虚线 = 随机）。单次运行，描述性结果。")


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
                "主指标为作者论文的 NS-AUC（`Lin's AUC`）。B2、B3、B4 为本项目方法，其余 11 个为作者发布的模型结果。")
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
                                                  range=["#087E78", "#B9C2C7"])),
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
    "**基准不是 SOTA**：纯表达反转（B1）NS-AUC 0.482，约等于随机；对齐指标方向并集成 BNNR 的 B4 在随机拆分"
    "排第 1/13，但与 BNNR 的差异不显著（校正检验 p = 0.066），弱相关拆分排第 3/13。提分来自已知关联结构，不是反转信号。",
    "**weakly correlated 划分**在 100 个种子中重复同一个外部测试集，区间反映内部折/模型选择波动，而非 100 次独立划分。",
    "**LUAD Top-10 全部为 insufficient_evidence**：9/10 为糖皮质激素类同一类别信号；仅 2 个与 Broad Hub InChIKey 完全一致；"
    "没有候选特异的 LUAD 动物或临床疗效证据，也没有湿实验数据。",
    "**文献检索有边界**：“未找到研究”只表示在限定检索中未找到，不代表不存在。",
    "**LLM 结论范围有限**：路由评测（v3 holdout：DeepSeek 98/100，规则 90/100）只衡量工具选择；"
    "两次模型选择不能证明一般化能力；五分区方法选择实验为负结果（0.40653 vs 固定 B2 0.51684）。",
    "**安全边界**：处方、剂量、患者建议等请求一律转人工审核；Agent 不给出治疗推荐。",
    "**只支持登记的 5 个疾病**（肺腺癌、乳腺癌、结直肠癌、前列腺癌、黑色素瘤）：上限取决于药物端有无匹配的 LINCS 细胞系；"
    "未登记的疾病转人工。参考药恢复只在乳腺癌显著，前列腺癌比随机还差，结直肠癌无法评估——流程可推广，不代表预测有效。",
    "**文献审阅**：除肺腺癌外没有冻结审阅，实时审阅的分级未经人工复核；乳腺癌与黑色素瘤分别有 23/62、20/78 条引语被逐字校验拒绝。",
]


def tab_limitations(st) -> None:
    st.markdown("\n".join(f"{i}. {text}" for i, text in enumerate(KEY_LIMITATIONS, start=1)))
    with st.expander("docs/limitations.md 原文"):
        st.markdown("\n".join(f"- {x}" for x in dd.load_limitations()))


def main() -> None:
    import streamlit as st

    os.chdir(dd.ROOT)  # package code resolves data/ and artifacts/ relative to the repo root
    st.set_page_config(page_title="药物重定位 Agent 演示", page_icon="💊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("可审计的药物重定位 Agent")
    st.markdown('<div class="hero-sub">自然语言请求 → 大模型多步规划 → 代码校验 → 真实工具执行 → 全程可追溯 · '
                '仅供研究，不构成治疗建议</div>', unsafe_allow_html=True)
    kpi_cards(st)
    tabs = st.tabs(["Agent 运行", "疾病登记表", "LUAD 候选", "Benchmark", "局限性", "Agent v1（旧版）"])
    with tabs[0]:
        tab_agent_v2(st)
    with tabs[1]:
        tab_registry(st)
    with tabs[2]:
        tab_luad(st)
    with tabs[3]:
        tab_benchmark(st)
    with tabs[4]:
        tab_limitations(st)
    with tabs[5]:
        tab_agent(st)


if __name__ == "__main__":
    main()
