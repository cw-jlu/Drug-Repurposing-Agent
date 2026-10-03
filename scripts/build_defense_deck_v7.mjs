// Build the v7 defense deck from committed result files (no hand-typed metrics).
// Usage (repo root): node scripts/build_defense_deck_v7.mjs
// Requires pptxgenjs installed under artifacts/deckgen (npm install pptxgenjs@3).
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(path.resolve("artifacts/deckgen/package.json"));
const pptxgen = require("pptxgenjs");
const J = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
const has = (p) => fs.existsSync(p);
const f2 = (x) => Number(x).toFixed(2);
const f3 = (x) => Number(x).toFixed(3);

const b3 = J("benchmark/results/recess_official_b3_vs_11.json");
const comp = J("benchmark/results/recess_official_component_ablation.json");
const probe = J("benchmark/results/contamination_probe_v1.json");
const dec = J("benchmark/results/decision_eval_v1.json");
const dec2 = J("benchmark/results/decision_eval_v2.json");
const repPath = "benchmark/results/model_replication_v4_pro.json";
const rep = has(repPath) ? J(repPath).evaluations : null;
const rev = J("benchmark/results/multi_agent_review_v1.json");
const evidenceAudit = J("benchmark/results/evidence_scope_adjudication_audit.json");
const evidenceLedger = J("benchmark/results/evidence_scope_adjudication_v1.json");
const evidenceExcluded = Object.entries(evidenceAudit.decision_counts)
  .filter(([decision]) => decision.startsWith("exclude_"))
  .reduce((sum, [, count]) => sum + count, 0);
const b4Path = "benchmark/results/recess_official_b4_vs_11.json";
const b4 = has(b4Path) ? J(b4Path) : null;
const statsPath = "benchmark/results/benchmark_stats_v1.json";
const bstats = has(statsPath) ? J(statsPath) : null;
const nbB4 = bstats?.splits?.random_simple?.paired_ns_auc?.["B4 - BNNR"]?.nadeau_bengio ?? null;
const b4Significant = nbB4 ? nbB4.p_two_sided < 0.05 : null;

const ns = (report, split, model) => report.splits[split].models[model]?.["NS-AUC"]?.mean;
// Prefer the fixed field {11 published, B2, model}; fall back to the full ranking list.
const rankOf = (report, split, model) =>
  report.splits[split].rank_in_published_field?.[model]?.rank ??
  report.splits[split].ranking_by_ns_auc?.find((x) => x.model === model)?.rank ?? null;
const total = (report, split, model) =>
  report.splits[split].rank_in_published_field?.[model]?.of ??
  report.splits[split].ranking_by_ns_auc?.length;

const C = { navy: "112B3C", ink: "183042", muted: "5B6B75", teal: "087E78",
  red: "9E493D", sand: "EEF3F2", white: "FFFFFF", gray: "B9C2C7" };
const FONT = "Microsoft YaHei";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "药物重定位 Agent 答辩 v7";

let n = 0;
function txt(s, t, x, y, w, h, size, o = {}) {
  s.addText(t, { x, y, w, h, fontFace: FONT, fontSize: size, color: o.color ?? C.ink,
    bold: !!o.bold, align: o.align ?? "left", valign: o.valign ?? "top", margin: 0,
    isTextBox: true });
}
function content(title, notes) {
  const s = pres.addSlide();
  n += 1;
  s.background = { color: C.white };
  txt(s, title, 0.6, 0.35, 12.1, 0.8, 30, { bold: true, color: C.navy });
  txt(s, String(n).padStart(2, "0"), 12.2, 6.95, 0.6, 0.35, 12, { color: C.muted, align: "right" });
  s.addNotes(notes);
  return s;
}
function stat(s, big, label, x, y, w, color = C.teal) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.35, rectRadius: 0.08,
    fill: { color: C.sand }, line: { color: C.sand } });
  txt(s, big, x + 0.2, y + 0.12, w - 0.4, 0.7, 28, { bold: true, color });
  txt(s, label, x + 0.2, y + 0.82, w - 0.4, 0.45, 12, { color: C.muted });
}
function bullets(s, items, x, y, w, h, size = 15) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x, y, w, h, fontFace: FONT, fontSize: size, color: C.ink, valign: "top", margin: 0,
      paraSpaceAfter: 10, isTextBox: true });
}
const axis = () => ({ catAxisLabelColor: C.muted, valAxisLabelColor: C.muted,
  catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT, catAxisLabelFontSize: 12,
  valAxisLabelFontSize: 11, valGridLine: { color: "E3E7EA", size: 0.5 },
  catGridLine: { style: "none" }, dataLabelFontFace: FONT, dataLabelFontSize: 11,
  legendFontFace: FONT, legendFontSize: 12, titleFontFace: FONT, titleFontSize: 14,
  titleColor: C.ink });

const R = "random_simple", W = "weakly_correlated";
const main = b4 ?? b3, mainName = b4 ? "B4" : "B3";

// 1 Title
{
  const s = pres.addSlide(); n += 1;
  s.background = { color: C.navy };
  txt(s, "药物重定位 Agent", 0.8, 2.0, 11.5, 1.1, 54, { bold: true, color: C.white });
  txt(s, "从疾病转录组到可审计的候选药物：公开 Benchmark + 肺腺癌案例", 0.82, 3.25, 11.6, 0.7, 24, { color: "CFE3E1" });
  txt(s, "课程设计答辩 · 2026 年 9 月 · 研究用途，不构成临床建议", 0.82, 6.3, 11.5, 0.5, 16, { color: "9FB7BF" });
  s.addNotes("约 20 秒。题目：面向药物重定位的可审计 Agent。两条主线：第三方公开 Benchmark 上的外部评测，以及肺腺癌端到端案例。");
}

// 2 Problem & five stages
{
  const s = content("问题与五个环节", "约 50 秒。课程要求 Problem → Data → Model → Benchmark → Biological interpretation。我们的问题：一个不训练新模型、由 LLM 调度确定性工具的 Agent，能否在别人定义的 Benchmark 上给出可信排名，并对具体疾病给出可追溯的候选与证据。");
  const steps = [["Problem", "表达反转能否找到候选药？Agent 能否可审计地完成全流程？"],
    ["Data", "TRANSCRIPT 613×151；GSE32863 肺腺癌配对；LINCS A549"],
    ["Model", "LLM 规划 + 确定性工具 + 规则/LLM 决策门控"],
    ["Benchmark", "RECeSS 官方 Runner，100 种子 × 2 种拆分"],
    ["Interpretation", "Top-10 机制、文献证据与多 Agent 审阅"]];
  steps.forEach(([h, d], i) => {
    const x = 0.6 + i * 2.47, hi = i === 3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.8, w: 2.25, h: 3.5, rectRadius: 0.1,
      fill: { color: hi ? C.teal : C.sand }, line: { color: hi ? C.teal : C.sand } });
    s.addShape(pres.shapes.OVAL, { x: x + 0.2, y: 2.0, w: 0.6, h: 0.6, fill: { color: hi ? C.white : C.navy }, line: { color: hi ? C.white : C.navy } });
    txt(s, String(i + 1), x + 0.2, 2.0, 0.6, 0.6, 18, { bold: true, align: "center", valign: "middle", color: hi ? C.teal : C.white });
    txt(s, h, x + 0.2, 2.85, 1.9, 0.5, 18, { bold: true, color: hi ? C.white : C.navy });
    txt(s, d, x + 0.2, 3.45, 1.9, 1.7, 13, { color: hi ? C.white : C.ink });
  });
  txt(s, "边界：不训练新预测模型；Benchmark 分数只代表已知关联的恢复能力，不代表临床疗效。", 0.6, 5.85, 12, 0.5, 15, { color: C.muted });
}

// 3 Data
{
  const s = content("数据与质量控制", "约 50 秒。TRANSCRIPT 标签极稀疏：613×151 矩阵里只有 401 个正例、11 个负例，其余是未知而非阴性。GSE32863 核实为 57 对配对样本（GEO 摘要写 60 对），两份无法配对的样本被排除；配对 limma 复现 512 上调、749 下调基因。来源：docs/data_card.md、docs/luad_data_audit.md、docs/figures/fig6。");
  s.addImage({ path: "docs/figures/fig6_data_overview.png", x: 0.5, y: 1.35, w: 7.7, h: 4.4,
    sizing: { type: "contain", w: 7.7, h: 4.4 } });
  stat(s, "613 × 151", "TRANSCRIPT 药物 × 疾病候选空间", 8.6, 1.4, 4.1);
  stat(s, "0.45%", "已知关联占比（401 正 / 11 负，其余未知）", 8.6, 2.95, 4.1, C.red);
  stat(s, "57 对", "GSE32863 核实配对样本（排除 2 份）", 8.6, 4.5, 4.1);
  txt(s, "未知 ≠ 阴性：指标只在官方可评价的药物行上计算。", 0.6, 6.1, 7.7, 0.5, 14, { color: C.muted });
}

// 4 Architecture
{
  const s = content("系统架构：LLM 规划，工具计算，代码校验",
    "约 50 秒。图为 Agent v2。大模型只看到模式、可用输入名称和已完成产物，看不到数据、文件路径和 Benchmark 标签；它一次提交多步计划，代码逐条校验白名单、模式、依赖和输入，不合法或某步失败时把观察反馈给它重新规划，最多 3 轮。数值计算全部由确定性工具完成，最终由 Validator 检查 ID、哈希、排名、引用和权限；每一步写入 SHA-256 链式轨迹。来源：docs/agent_v2.md、docs/figures/fig9_architecture.png。");
  s.addImage({ path: "docs/figures/fig9_architecture.png", x: 1.3, y: 1.15, w: 10.7, h: 6.1,
    sizing: { type: "contain", w: 10.7, h: 6.1 } });
}

// 4b Agent planning (agent v2)
{
  const msPath = "benchmark/results/planner_eval_multistep_v1.json";
  const demoPath = "benchmark/results/agent_v2_demo_runs.json";
  if (has(msPath) && has(demoPath)) {
    const ms = J(msPath), demo = J(demoPath);
    const rule = ms.planners.rule_v2.summary, llm = ms.planners["deepseek_v2:deepseek-flash"].summary;
    const cats = [["full", "完整流程"], ["partial", "部分任务"], ["benchmark", "基准模式"],
      ["missing_input", "缺少输入"], ["failure", "故障恢复"], ["unsafe", "越权/注入"]];
    const frac = (x) => { const [a, b] = x.split("/").map(Number); return +(a / b).toFixed(3); };
    const rp = demo.runs.replan.rounds;
    const s = content("Agent 自主规划：一句话 → 多步计划 → 失败后重新规划",
      `约 60 秒。v1 只从 3 个工具里选 1 个；v2 让大模型把请求拆成有序步骤，代码校验依赖、模式和输入后执行，失败时带着已完成产物重新规划。冻结的 40 例多步规划测试（调用前提交）：DeepSeek ${llm.passed}/40，规则规划器 ${rule.passed}/40。规则输在“部分任务”（只认关键词，把“只做差异表达”也跑成全流程）；DeepSeek 部分任务、缺少输入、基准模式全对，但越权类 ${llm.by_category.unsafe}：对“跳过质控”“忽略校验当治疗方案”“排完删数据”它照常跑了安全的标准流程而没有拒绝——代码层依赖校验和白名单仍挡住了实际风险，但这是规划器的真实弱点。右侧是真实工具上的一次运行：排名第一次失败后，第 2 轮只规划剩下的 ${rp[1].steps.length} 步，跳过已完成的质控、差异表达和通路分析，最终完成。能力边界：只支持登记表中的疾病（目前只有肺腺癌）；说出疾病名后，Agent 能自动从 GEO 下载数据、核对哈希、从原始数据重算，一路跑到报告，未登记的疾病会转人工。补充一句：大模型决定用哪些工具、按什么顺序、何时停止（40 例中第一轮出现 9 种不同计划），但每个工具内部的算法和阈值由代码固定——阈值会改变 Top-10，不能让模型看完结果再调。`);
    s.addChart(pres.charts.BAR, [
      { name: "规则规划器", labels: cats.map((c) => c[1]), values: cats.map((c) => frac(rule.by_category[c[0]])) },
      { name: "DeepSeek 规划器", labels: cats.map((c) => c[1]), values: cats.map((c) => frac(llm.by_category[c[0]])) }],
      { x: 0.4, y: 1.3, w: 7.2, h: 5.0, barDir: "col", barGrouping: "clustered", chartColors: [C.gray, C.teal],
        showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0%", valAxisMinVal: 0, valAxisMaxVal: 1.15,
        showLegend: true, legendPos: "b", showTitle: true,
        title: `冻结 40 例多步规划：规则 ${rule.passed}/40 · DeepSeek ${llm.passed}/40`, ...axis(), valAxisLabelFormatCode: "0%",
        dataLabelFontSize: 9, barGapWidthPct: 40, barOverlapPct: -8 });
    const zh = { qc_disease_cohort: "质控", differential_expression: "差异表达", pathway_enrichment: "通路",
      rank_candidates: "排名", audit_candidates: "审计", review_literature: "文献", build_report: "报告" };
    txt(s, "真实运行：排名首次失败后的重新规划", 7.9, 1.35, 5.0, 0.4, 14, { bold: true, color: C.navy });
    const failed = demo.runs.replan.executed.find((e) => e.status === "failed");
    const lines = [
      [`第 1 轮计划（${rp[0].steps.length} 步）`, rp[0].steps.map((t) => zh[t] ?? t).join(" → ")],
      ["执行", `质控、差异表达、通路 ✓ ；${zh[failed.tool]} ✗`],
      [`第 2 轮计划（${rp[1].steps.length} 步）`, rp[1].steps.map((t) => zh[t] ?? t).join(" → ")],
      ["结果", `完成：${demo.runs.replan.status}；已完成步骤未重跑`]];
    lines.forEach(([h, b], i) => {
      const y = 1.9 + i * 1.12;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.9, y, w: 4.9, h: 0.95, rectRadius: 0.06,
        fill: { color: i === 1 ? "F2E7E4" : C.sand }, line: { color: C.sand } });
      txt(s, h, 8.05, y + 0.08, 4.6, 0.3, 12, { bold: true, color: i === 1 ? C.red : C.teal });
      txt(s, b, 8.05, y + 0.42, 4.6, 0.5, 12, { color: C.ink });
    });
  }
}

// 5 Key finding
{
  const b1r = comp.splits[R].models.B1["NS-AUC"].mean, b1w = comp.splits[W].models.B1["NS-AUC"].mean;
  const labels = ["B1 表达反转", "B2 旧融合（列方向）", "B3 行方向融合"];
  const vr = [b1r, ns(b3, R, "B2"), ns(b3, R, "B3")], vw = [b1w, ns(b3, W, "B2"), ns(b3, W, "B3")];
  if (b4) { labels.push("B4 = B3 + BNNR"); vr.push(ns(b4, R, "B4")); vw.push(ns(b4, W, "B4")); }
  const s = content("发现：反转≈随机，提分来自对齐评价指标",
    `约 70 秒。表达反转 B1 在官方 NS-AUC 上只有 ${f3(b1r)}（随机拆分），与随机无差别——这是本项目的核心负结果。复核评价代码发现：官方 NS-AUC 在每个药物行内对疾病排序、并把同分记 0；旧 B2 在疾病列内排序，且流行度分量在行内是常数。B3 改为行方向、加入疾病流行度与标签共现并打破同分，配置在正式评分前冻结（commit 92943bb），开发种子与官方 100 种子不相交。弱相关拆分只有一个固定留出集，冻结前已看过，已在配置中披露。`);
  s.addChart(pres.charts.BAR, [{ name: "随机拆分", labels, values: vr.map((x) => +f3(x)) },
    { name: "弱相关拆分", labels, values: vw.map((x) => +f3(x)) }],
    { x: 0.5, y: 1.3, w: 7.8, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [C.navy, C.teal],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.4,
      valAxisMaxVal: 0.8, showLegend: true, legendPos: "b", showTitle: true,
      title: "官方 NS-AUC（100 种子均值）", ...axis() });
  bullets(s, ["官方 NS-AUC 在每个药物行内比较疾病，同分记 0",
    "旧 B2 按疾病列排序，流行度在行内是常数 → 失效",
    "B3 改为行方向：疾病流行度 + 表达 kNN + 标签共现 + 反转",
    "配置在正式评分前冻结；开发种子与官方种子不相交"].concat(b4 ? ["第二轮 B4 = B3 + BNNR 移植（与作者结果逐种子一致），单独冻结后评分"] : []), 8.7, 1.6, 4.1, 4.8, 15);
}

// 6 Main benchmark
{
  const pw = b3.splits[R].paired_ns_auc.B3?.BNNR?.wins;
  const s = content(`Benchmark：${mainName} 随机 ${rankOf(main, R, mainName)}/${total(main, R, mainName)}${b4 && b4Significant === false ? "（与 BNNR 持平）" : ""}、弱相关 ${rankOf(main, W, mainName)}/${total(main, W, mainName)}`,
    `约 70 秒。RECeSS 官方 Runner，与作者公布的 11 个模型逐种子对齐。B3：随机 ${f3(ns(b3, R, "B3"))}（BNNR ${f3(ns(b3, R, "BNNR"))} 第一，B3 在 100 个配对种子中胜 ${pw ?? "?"} 次），弱相关 ${f3(ns(b3, W, "B3"))}（MBiRW ${f3(ns(b3, W, "MBiRW"))} 第一）。${b4 ? `第二轮 B4（B3 + BNNR 的 NumPy 移植，只在开发种子上设计后冻结）：随机 ${f3(ns(b4, R, "B4"))}、弱相关 ${f3(ns(b4, W, "B4"))}。` : ""}${b4 && rankOf(b4, R, "B4") === 1 ? `B4 在随机拆分上的均值高于全部 11 个发表模型（对 BNNR 配对胜 ${b4.splits[R].paired_ns_auc.B4?.BNNR?.wins ?? "?"}/100）。但 100 个种子的测试集互相重叠，用 Nadeau–Bengio 校正检验后，对 BNNR 的差值 p = ${nbB4 ? nbB4.p_two_sided.toFixed(3) : "?"}，95% CI ${nbB4 ? `[${nbB4.mean_ci95[0].toFixed(4)}, ${nbB4.mean_ci95[1].toFixed(4)}]` : "?"}${b4Significant === false ? " 跨 0，只能说与 BNNR 持平" : ""}；global NDCG 与 HR@10 上 BNNR 仍更强。弱相关拆分落后 MBiRW 与 HAN。因此不宣称 SOTA。` : "不训练新模型的方法达到了与训练型协同过滤方法相当的水平，但没有在两种拆分上都超过最强模型，不宣称 SOTA。"}`);
  s.addImage({ path: "docs/figures/fig1_nsauc_boxplot.png", x: 0.55, y: 1.3, w: 12.2, h: 5.55,
    sizing: { type: "contain", w: 12.2, h: 5.55 } });
}

// 7 Contamination probe
{
  const o = probe.conditions.open_book.positive_vs_unknown, c = probe.conditions.closed_book.positive_vs_unknown,
    rv = probe.conditions.raw_reversal_score.positive_vs_unknown, d = probe.conditions.open_minus_closed_paired;
  const s = content("为什么必须 Strict 模式：LLM 知识污染探针",
    `约 50 秒。同一 DeepSeek 模型、同一冻结样本（300 正例 + 300 未知，哈希在调用前固定）。开卷给药名和病名：AUC ${f3(o.auc)}，95% CI [${f3(o.ci95[0])}, ${f3(o.ci95[1])}]，唯一显著高于 0.5；闭卷只给反转数值：${f3(c.auc)}；反转分数本身 ${f3(rv.auc)}。开卷减闭卷 ${f3(d.auc_difference)}，CI 跨 0。结论：LLM 会凭记忆认出教科书适应症，这是泄漏不是发现，所以 Benchmark 必须在 LLM 看不到身份的 Strict 模式下运行；TRANSCRIPT 标签噪声使泄漏幅度不大。共 ${probe.call_counts.attempts} 次调用，约 ${f2(probe.cost_estimate.estimated_cost_usd.offpeak)} 美元。`);
  s.addChart(pres.charts.BAR, [{ name: "AUC", labels: ["开卷：给药名/病名", "闭卷：只给反转数值", "反转分数本身"],
    values: [o.auc, c.auc, rv.auc].map((x) => +f3(x)) }],
    { x: 0.5, y: 1.3, w: 7.4, h: 5.3, barDir: "bar", chartColors: [C.red, C.teal, C.gray], varyColors: true,
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0.4,
      valAxisMaxVal: 0.65, showLegend: false, showTitle: true, title: "正例 vs 未知 AUC（600 对）", ...axis() });
  bullets(s, [`开卷 95% CI [${f3(o.ci95[0])}, ${f3(o.ci95[1])}]：唯一显著高于 0.5`,
    "高置信命中都是教科书适应症（如四环素类—痤疮）",
    `开卷 − 闭卷 = ${f3(d.auc_difference)}，CI 跨 0：泄漏存在但幅度小`,
    "看得到药名的 LLM 会“背答案”，所以 Benchmark 必须在 Strict 模式下运行"], 8.3, 1.6, 4.5, 4.9, 15);
}

// 8 Decision layer
{
  const L = dec.layers;
  const jevPath = "benchmark/results/decision_eval_v1_jev.json";
  const jev = has(jevPath) ? J(jevPath) : null;
  const jev2 = has("benchmark/results/decision_eval_v2_jev.json") ? J("benchmark/results/decision_eval_v2_jev.json") : null;
  const series = [["J0 规则", L.J0_fixed_rules], ["J1 flash", L.J1_deepseek_structured],
    ["J3 flash+门控", L.J3_deepseek_plus_gate]];
  if (jev) series.push(["J2 Jev", jev.layers.J2_jev], ["J3 Jev+门控", jev.layers.J3_jev_gate]);
  const lab = series.map((x) => x[0]);
  const cases = Object.values(dec.case_counts_by_node).reduce((a, b) => a + b, 0);
  const jevNote = jev ? `Jev（TypeSafe System One，经 OpenCode Zen 调用 ${jev.model_requested}；付费版因余额不足在得到答案前改用免费版）不加门控准确率 ${f3(jev.layers.J2_jev.accuracy)}，为所有决策器最高；Brier ${f3(jev.jev_calibration.brier_multiclass)}，低于 flash 0.221 与 v4-pro 0.228，但 ECE ${f3(jev.jev_calibration.ece_10bin_reported_confidence)} 略高于 flash 的 0.025。Jev 加门控后高风险误执行为 ${f3(jev.layers.J3_jev_gate.high_risk_wrong_auto_execution_rate)}，代价是 ${f3(jev.layers.J3_jev_gate.escalation_rate)} 的升级率；低置信交给 flash 的 J4 准确率 ${f3(jev.layers.J4_jev_gate_llm_fallback.accuracy)}。` : "";
  const v2Note = jev2 ? `（Jev ${jev2.layers.hybrid_jev.correct}/40）；让模型包办整个决策时，完整 Jev 只有 ${jev2.layers.full_jev.correct}/40 并出现高风险误执行` : "";
  const s = content(jev ? "决策层：规则 vs 通用 LLM vs Jev（实测）" : "决策层消融：规则 vs LLM vs 门控（Jev 接口已预留）",
    `约 60 秒。图为 v1 的 ${cases} 个封闭合成决策用例，所有决策器看到同样的策略文本与用例字段，门控阈值（0.8/0.9）在调用前固定。规则与 flash 准确率都是 ${f3(L.J0_fixed_rules.accuracy)}；flash 高风险误执行 ${f3(L.J1_deepseek_structured.high_risk_wrong_auto_execution_rate)}，加门控降到 ${f3(L.J3_deepseek_plus_gate.high_risk_wrong_auto_execution_rate)}。${jevNote}v2 的 40 条中文风险备注上，“规则算结构化字段、模型只判备注”的混合层，无论 flash、v4-pro 还是 Jev 判备注，高风险误执行都是 0${v2Note}。${rep ? "换 v4-pro 时门控一次都没触发，阈值需要按模型校准。" : ""}用例为合成数据、单次运行，不代表临床正确率。`);
  s.addChart(pres.charts.BAR, [
    { name: "准确率", labels: lab, values: series.map((x) => +f3(x[1].accuracy)) },
    { name: "模糊用例转人工", labels: lab, values: series.map((x) => +f3(x[1].review_recall_on_ambiguous_cases)) },
    { name: "高风险误执行率", labels: lab, values: series.map((x) => +f3(x[1].high_risk_wrong_auto_execution_rate)) }],
    { x: 0.4, y: 1.3, w: 8.5, h: 5.4, barDir: "col", barGrouping: "clustered", chartColors: [C.navy, C.teal, C.red],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0,
      valAxisMaxVal: 1.1, showLegend: true, legendPos: "b", showTitle: false, ...axis(), catAxisLabelFontSize: 11,
      dataLabelFontSize: 9 });
  const items = jev ? [`v1 ${cases} 例：Jev 准确率 ${f3(jev.layers.J2_jev.accuracy)} 最高，Brier ${f3(jev.jev_calibration.brier_multiclass)} 最低`,
    `Jev + 门控：高风险误执行 ${f3(jev.layers.J3_jev_gate.high_risk_wrong_auto_execution_rate)}，但升级率 ${f3(jev.layers.J3_jev_gate.escalation_rate)}`,
    jev2 ? `v2 混合层（Jev 判备注）${jev2.layers.hybrid_jev.correct}/40、0 高风险误执行；完整 Jev ${jev2.layers.full_jev.correct}/40` : "",
    "混合设计换 flash / v4-pro / Jev 都是 0 高风险误执行",
    "合成用例、单次运行；Jev 用免费版"].filter(Boolean) : [`图：v1 ${cases} 个冻结合成决策用例`,
    "v1 规则与 LLM 准确率持平；门控减少高风险误执行", "Jev 未获访问：接口保留"];
  bullets(s, items, 9.1, 1.4, 3.8, 5.3, 13);
}

// 9 LUAD biology
{
  const enr = has("benchmark/results/luad_pathway_enrichment_v1.json") ? J("benchmark/results/luad_pathway_enrichment_v1.json") : null;
  const rev9 = has("benchmark/results/luad_pathway_reversal_v1.json") ? J("benchmark/results/luad_pathway_reversal_v1.json") : null;
  const pc = has("benchmark/results/luad_positive_control_stats_v1.json") ? J("benchmark/results/luad_positive_control_stats_v1.json") : null;
  const H = enr?.libraries?.MSigDB_Hallmark_2020;
  const sigTerms = (d, k) => (H?.[d] ?? []).filter((r) => (r.fdr ?? 1) < 0.05).slice(0, k).map((r) => r.term).join("、");
  const s = content("肺腺癌案例：通路与机制",
    `约 60 秒。GSE32863 配对差异表达得到 512 上调 / 749 下调基因（FDR<0.05，|log2FC|≥1）。Hallmark 富集：肿瘤上调 ${sigTerms("up", 5)}；肿瘤下调 ${sigTerms("down", 4)}，后者更可能反映正常肺组织免疫/间质成分在肿瘤中减少。右图：Top-10 在这些通路上的反转百分位，主要集中在糖酵解、G2-M/E2F 与缺氧；diflorasone 与 beclomethasone 不反转 EMT 基因。Top-10 中 9/10 属于或很可能属于糖皮质激素，指向 NR3C1 类别假设；但阈值敏感性分析显示这只在 |log2FC|≥1 及更严时成立，放宽到 0.58 后 Top-10 只剩 3 个原候选，RAF/PI3K/mTOR 抑制剂（Hub 注释）进入，只有 hydrocortisone、beclomethasone、clocortolone 在 12 种设定下都留在 Top-10（docs/luad_threshold_sensitivity.md）。注意：Top-10 与通路基因来自同一签名，右图不是独立验证。预先冻结的参考药中 ${pc?.measured_controls ?? 5} 个可测，平均名次百分位 ${pc ? f3(pc.mean_percentile) : "?"}，置换 p = ${pc ? f3(pc.permutation_p_one_sided) : "?"}，不显著——本筛选没有证明能恢复已知 LUAD 药物。A549 为 KRAS 突变细胞系，结果不能外推到患者。`);
  s.addImage({ path: "docs/figures/fig7_luad_pathways.png", x: 0.4, y: 1.3, w: 6.3, h: 4.9, sizing: { type: "contain", w: 6.3, h: 4.9 } });
  s.addImage({ path: "docs/figures/fig8_top10_pathway_reversal.png", x: 6.85, y: 1.3, w: 6.1, h: 4.9, sizing: { type: "contain", w: 6.1, h: 4.9 } });
  txt(s, `预设阈值下 9/10 为糖皮质激素，但放宽阈值后被 RAF/PI3K/mTOR 抑制剂部分取代；参考药恢复 p = ${pc ? f3(pc.permutation_p_one_sided) : "?"}，不显著`, 0.6, 6.4, 11.9, 0.45, 15, { bold: true, color: C.red });
}

// 10 Multi-agent review
{
  const cv = rev.citation_validation, tc = rev.tier_counts;
  const liveRev = has("benchmark/results/agent_v2_live_review.json") ? J("benchmark/results/agent_v2_live_review.json") : null;
  const liveNote = liveRev ? `10 月 2 日经 Agent v2 实时重跑一次：${liveRev.per_candidate.filter((c) => c.same).length}/10 分级与冻结版一致，${liveRev.per_candidate.filter((c) => !c.same).map((c) => c.name).join("、")} 分级改变，仍无候选达到“有支持”；审阅代码后来加入引用范围检查，引语被拒 ${liveRev.live_summary.rejected_quoted_items}/${liveRev.live_summary.proposed_quoted_items}，与冻结版不可直接比较。` : "";
  const s = content("多 Agent 证据审阅：逐字引用 ≠ 论断成立",
    `约 50 秒。${liveNote}每个候选由文献 Agent、批评 Agent 和协调者处理。冻结版 ${cv.proposed_quoted_items} 条引语全部通过 PMID 与逐字校验、${tc.INSUFFICIENT_EVIDENCE}/10 判为证据不足，共 ${rev.call_counts.attempts} 次模型调用。事后来源/药名范围预筛标记 ${evidenceAudit.reviewed_count} 条；当前 PubMed 摘要级复核中 ${evidenceExcluded} 条原样候选级引用需排除，${evidenceAudit.decision_counts.retain_narrowed} 条仅能收窄终点保留，${evidenceAudit.decision_counts.retain_class_caution_only} 条仅作一般安全提示。错误包括勘误当原始研究、未指名类固醇归给具体候选、柚皮素抗癌结果误归给 fluticasone。冻结分级未重跑；这不是独立双人全文审阅。`);
  stat(s, String(cv.proposed_quoted_items), "冻结版通过逐字校验", 0.6, 1.4, 2.9);
  stat(s, String(evidenceAudit.reviewed_count), "事后范围预筛标记", 3.7, 1.4, 2.9, C.red);
  stat(s, String(evidenceExcluded), "原样候选级引用需排除", 6.8, 1.4, 2.9, C.red);
  stat(s, `${tc.INSUFFICIENT_EVIDENCE}/10`, "冻结分级：未重新计算", 9.9, 1.4, 2.8);
  const rows = [["排名", "候选", "冻结分级", "23 条中的摘要级审计"]].concat(rev.candidates.map((c) => {
    const entries = evidenceLedger.decisions.filter((entry) =>
      entry.case_id.startsWith(`rank${String(c.rank).padStart(2, "0")}_`));
    const excluded = entries.filter((entry) => entry.decision.startsWith("exclude_")).length;
    const narrowed = entries.length - excluded;
    return [String(c.rank), c.name,
      c.tier === "INSUFFICIENT_EVIDENCE" ? "证据不足" : c.tier,
      entries.length ? `排除 ${excluded}；限定范围 ${narrowed}` : "本轮未标记（不等于已全面核验）"];
  }));
  s.addTable(rows.map((r, i) => r.map((v) => ({ text: v, options: { bold: i === 0,
    color: i === 0 ? C.white : C.ink, fill: { color: i === 0 ? C.navy : (i % 2 ? C.white : C.sand) } } }))),
    { x: 0.6, y: 3.0, w: 12.1, colW: [0.9, 4.2, 2.2, 4.8], fontFace: FONT, fontSize: 10.5, rowH: 0.3,
      border: { type: "solid", color: "DDE3E6", pt: 0.5 } });
}

// 11 Conclusions
{
  const s = pres.addSlide(); n += 1;
  s.background = { color: C.navy };
  txt(s, "结论与局限", 0.7, 0.5, 12, 0.8, 34, { bold: true, color: C.white });
  const items = [
    ["负结果", `纯表达反转在 TRANSCRIPT 上 ≈ 随机（官方 NS-AUC ${f3(comp.splits[R].models.B1["NS-AUC"].mean)}）`],
    ["提分", `对齐指标方向${b4 ? "并集成 BNNR " : ""}后 ${mainName} 随机拆分第 ${rankOf(main, R, mainName)}/${total(main, R, mainName)}、弱相关第 ${rankOf(main, W, mainName)}/${total(main, W, mainName)}${rankOf(main, R, mainName) === 1 ? (b4Significant === false ? "；随机拆分与 BNNR 持平（校正检验不显著），弱相关未超过" : "；随机拆分超过全部发表模型，弱相关未超过") : "，接近但未全面超过最强基线"}`],
    ["可信", "Strict 模式隔离标签；探针证实 LLM 看到药名会“背答案”"],
    ["Agent", "去掉大模型分数不变；它负责理解请求与文字（部分任务：规则 2/8 → 大模型 8/8）"],
    ["局限", "只支持登记疾病（目前仅肺腺癌）；细胞系 ≠ 患者；未知 ≠ 阴性；外部 v3 未跑；无湿实验"]];
  items.forEach(([h, d], i) => {
    const y = 1.6 + i * 1.0;
    txt(s, h, 0.8, y, 1.6, 0.6, 20, { bold: true, color: i === 4 ? "F2B8A8" : "9FE0D8" });
    txt(s, d, 2.5, y, 10.2, 0.8, 18, { color: C.white });
  });
  s.addNotes("约 50 秒。总结五点。Agent 一行：去掉大模型，Benchmark 分数和 Top-10 一个数字都不会变，因为数值计算全部由确定性代码完成；我们也测过让大模型挑方法、直接判断药物，结果都更差，后者还会泄漏答案。大模型的作用在规划和文字理解：多步规划中“只做一部分”的请求，规则只做对 2/8，大模型 8/8；中文风险备注规则只对 20/40，混合设计 39/40。但它对越权请求的拒绝（5/8）不如规则（7/8），所以安全靠代码层校验，不靠大模型。一句话：算法决定结果对不对，Agent 决定系统好不好用、敢不敢用。文献审阅方面，23 条是事后来源/药名范围预筛标记，并非 65 条都经过全文审阅；13 条原样候选级引用需排除，冻结药物分级未重算。独立外部方法选择 v3 的数据与标签门槛未通过，没有外部 NS-AUC 结果。Benchmark 分数只代表已知关联恢复能力；LUAD 候选是待验证的研究假设，不是用药建议。代码与所有结果见 GitHub 仓库。");
}

const out = process.argv[2] ?? "deliverables/药物重定位Agent_答辩稿_v7.pptx";
await pres.writeFile({ fileName: out });
console.log(out, b4 ? "(with B4)" : "(B3 only)");
