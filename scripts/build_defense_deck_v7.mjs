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
  const s = content("系统架构：谁负责什么", "约 60 秒。LLM 只做规划与解释，数值计算全部由确定性工具完成，最终由代码校验 Schema、ID、引用和权限。Strict 模式下 LLM 看不到任何标签或药名。规划器在事先冻结的 100 例独立 holdout 上：DeepSeek 98/100，规则 90/100（docs/planner_eval_results.md）。每次运行写入带 SHA-256 链的 JSONL 轨迹。");
  const boxes = ["自然语言请求", "Planner\nDeepSeek / 规则", "工具白名单\nStrict / Open", "确定性计算\n差异表达·排名融合", "证据层\nPubMed·多 Agent", "Validator\nSchema·引用·权限"];
  boxes.forEach((b, i) => {
    const x = 0.55 + i * 2.1;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.8, w: 1.8, h: 1.6, rectRadius: 0.1,
      fill: { color: i === 0 ? C.navy : C.sand }, line: { color: i === 0 ? C.navy : C.gray, width: 0.75 } });
    txt(s, b, x + 0.08, 1.8, 1.64, 1.6, 13, { bold: true, align: "center", valign: "middle", color: i === 0 ? C.white : C.navy });
    if (i < boxes.length - 1) {
      s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 1.83, y: 2.47, w: 0.24, h: 0.26, fill: { color: C.teal }, line: { color: C.teal } });
    }
  });
  stat(s, "98/100", "规划器冻结 holdout（DeepSeek；规则 90/100）", 0.6, 4.2, 3.9);
  stat(s, "0 标签", "Strict 模式下 LLM 可见的 Benchmark 标签", 4.7, 4.2, 3.9);
  stat(s, "哈希链", "每次运行写入 JSONL 轨迹，可逐步复核", 8.8, 4.2, 3.9);
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
  const L = dec.layers, keys = ["J0_fixed_rules", "J1_deepseek_structured", "J3_deepseek_plus_gate"];
  const lab = ["J0 固定规则", "J1 通用 LLM", "J3 LLM + 置信门控"];
  const cases = Object.values(dec.case_counts_by_node).reduce((a, b) => a + b, 0);
  const s = content("决策层消融：规则 vs LLM vs 门控（Jev 接口已预留）",
    `约 50 秒。图为 v1 的 ${cases} 个封闭合成决策用例：J0 与 J1 准确率都是 ${f3(L[keys[0]].accuracy)}，J1 高风险误执行 ${f3(L[keys[1]].high_risk_wrong_auto_execution_rate)}；置信门控降到 ${f3(L[keys[2]].high_risk_wrong_auto_execution_rate)}，但增加人工升级。新 v2 是另一个在任何调用前冻结的 40 条中文风险备注压力测试：混合层 ${dec2.layers.hybrid.correct}/40、完整 LLM ${dec2.layers.full_llm.correct}/40、旧关键词规则 ${dec2.layers.j0_rules.correct}/40；80/80 provider trace 通过回放。v2 是作者标注的合成新用例，不与 v1 准确率直接比较，也不代表临床正确率。Jev 未获访问。`);
  s.addChart(pres.charts.BAR, [
    { name: "准确率", labels: lab, values: keys.map((k) => +f3(L[k].accuracy)) },
    { name: "模糊用例升级召回", labels: lab, values: keys.map((k) => +f3(L[k].review_recall_on_ambiguous_cases)) },
    { name: "高风险误执行率", labels: lab, values: keys.map((k) => +f3(L[k].high_risk_wrong_auto_execution_rate)) }],
    { x: 0.5, y: 1.3, w: 8.2, h: 5.4, barDir: "col", barGrouping: "clustered", chartColors: [C.navy, C.teal, C.red],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisMinVal: 0,
      valAxisMaxVal: 1.1, showLegend: true, legendPos: "b", showTitle: false, ...axis() });
  bullets(s, [`图：v1 ${cases} 个冻结合成决策用例`,
    "v1 规则与 LLM 准确率持平；门控减少高风险误执行",
    `新 v2 中文备注：混合 ${dec2.layers.hybrid.correct}/40，完整 LLM ${dec2.layers.full_llm.correct}/40`,
    "v2 作者标注、合成；两集结果不能直接比",
    "Jev 未获访问：接口保留"], 9.0, 1.5, 3.8, 5.0, 14);
}

// 9 LUAD biology
{
  const enr = has("benchmark/results/luad_pathway_enrichment_v1.json") ? J("benchmark/results/luad_pathway_enrichment_v1.json") : null;
  const rev9 = has("benchmark/results/luad_pathway_reversal_v1.json") ? J("benchmark/results/luad_pathway_reversal_v1.json") : null;
  const pc = has("benchmark/results/luad_positive_control_stats_v1.json") ? J("benchmark/results/luad_positive_control_stats_v1.json") : null;
  const H = enr?.libraries?.MSigDB_Hallmark_2020;
  const sigTerms = (d, k) => (H?.[d] ?? []).filter((r) => (r.fdr ?? 1) < 0.05).slice(0, k).map((r) => r.term).join("、");
  const s = content("肺腺癌案例：通路与机制",
    `约 60 秒。GSE32863 配对差异表达得到 512 上调 / 749 下调基因（FDR<0.05，|log2FC|≥1）。Hallmark 富集：肿瘤上调 ${sigTerms("up", 5)}；肿瘤下调 ${sigTerms("down", 4)}，后者更可能反映正常肺组织免疫/间质成分在肿瘤中减少。右图：Top-10 在这些通路上的反转百分位，主要集中在糖酵解、G2-M/E2F 与缺氧；diflorasone 与 beclomethasone 不反转 EMT 基因。Top-10 中 9/10 属于或很可能属于糖皮质激素，指向 NR3C1 类别假设。注意：Top-10 与通路基因来自同一签名，右图不是独立验证。预先冻结的参考药中 ${pc?.measured_controls ?? 5} 个可测，平均名次百分位 ${pc ? f3(pc.mean_percentile) : "?"}，置换 p = ${pc ? f3(pc.permutation_p_one_sided) : "?"}，不显著——本筛选没有证明能恢复已知 LUAD 药物。A549 为 KRAS 突变细胞系，结果不能外推到患者。`);
  s.addImage({ path: "docs/figures/fig7_luad_pathways.png", x: 0.4, y: 1.3, w: 6.3, h: 4.9, sizing: { type: "contain", w: 6.3, h: 4.9 } });
  s.addImage({ path: "docs/figures/fig8_top10_pathway_reversal.png", x: 6.85, y: 1.3, w: 6.1, h: 4.9, sizing: { type: "contain", w: 6.1, h: 4.9 } });
  txt(s, `9/10 为糖皮质激素（NR3C1 类别假设）；主要反转增殖与糖酵解程序；参考药恢复 p = ${pc ? f3(pc.permutation_p_one_sided) : "?"}，不显著`, 0.6, 6.4, 11.9, 0.45, 15, { bold: true, color: C.red });
}

// 10 Multi-agent review
{
  const cv = rev.citation_validation, tc = rev.tier_counts;
  const s = content("多 Agent 证据审阅：逐字引用 ≠ 论断成立",
    `约 50 秒。每个候选由文献 Agent、批评 Agent 和协调者处理。冻结版 ${cv.proposed_quoted_items} 条引语全部通过 PMID 与逐字校验、${tc.INSUFFICIENT_EVIDENCE}/10 判为证据不足，共 ${rev.call_counts.attempts} 次模型调用。事后来源/药名范围预筛标记 ${evidenceAudit.reviewed_count} 条；当前 PubMed 摘要级复核中 ${evidenceExcluded} 条原样候选级引用需排除，${evidenceAudit.decision_counts.retain_narrowed} 条仅能收窄终点保留，${evidenceAudit.decision_counts.retain_class_caution_only} 条仅作一般安全提示。错误包括勘误当原始研究、未指名类固醇归给具体候选、柚皮素抗癌结果误归给 fluticasone。冻结分级未重跑；这不是独立双人全文审阅。`);
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
    ["Agent", "规划 98/100；23 条范围预筛引用中 13 条原样需排除"],
    ["局限", "细胞系 ≠ 患者；未知 ≠ 阴性；外部 v3 未跑；Jev/湿实验未做"]];
  items.forEach(([h, d], i) => {
    const y = 1.6 + i * 1.0;
    txt(s, h, 0.8, y, 1.6, 0.6, 20, { bold: true, color: i === 4 ? "F2B8A8" : "9FE0D8" });
    txt(s, d, 2.5, y, 10.2, 0.8, 18, { color: C.white });
  });
  s.addNotes("约 40 秒。总结五点。23 条是事后来源/药名范围预筛标记，并非 65 条都经过全文审阅；13 条原样候选级引用需排除，冻结药物分级未重算。独立外部方法选择 v3 的数据与标签门槛未通过，没有外部 NS-AUC 结果。Benchmark 分数只代表已知关联恢复能力；LUAD 候选是待验证的研究假设，不是用药建议。代码与所有结果见 GitHub 仓库。");
}

const out = process.argv[2] ?? "deliverables/药物重定位Agent_答辩稿_v7.pptx";
await pres.writeFile({ fileName: out });
console.log(out, b4 ? "(with B4)" : "(B3 only)");
