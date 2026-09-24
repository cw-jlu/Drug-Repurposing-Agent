import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const skillDir = process.env.SKILL_DIR;
const runtimePython = process.env.RUNTIME_PYTHON;
const workspaceDir = process.cwd();
if (!path.isAbsolute(skillDir ?? "") || !path.isAbsolute(runtimePython ?? "")) {
  throw new Error("Set SKILL_DIR and RUNTIME_PYTHON to bundled absolute paths");
}
const { applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href,
);

const buildDir = path.join(workspaceDir, "artifacts", "slide_build");
const outputDir = path.join(workspaceDir, "deliverables");
await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(outputDir, { recursive: true });
const finalPath = path.join(outputDir, "药物重定位Agent_答辩草稿_v2.pptx");
const candidatePath = path.join(buildDir, "candidate-v2.pptx");

const pres = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slideList = [];
const font = "Microsoft YaHei";
const C = { navy: "#112B3C", ink: "#183042", muted: "#536672", teal: "#087E78",
  blue: "#4776A1", sand: "#F6F5F1", white: "#FFFFFF", red: "#9E493D" };

function text(slide, value, x, y, w, h, size, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox", position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = { typeface: font, fontSize: size,
    bold: Boolean(opts.bold), color: opts.color ?? C.ink, autoFit: "none" };
  return shape;
}

function slide(title, number, notes) {
  const s = pres.slides.add();
  slideList.push(s);
  s.background.fill = C.sand;
  text(s, title, 72, 42, 1136, 68, 43, { bold: true, color: C.navy });
  text(s, String(number).padStart(2, "0"), 1136, 654, 72, 30, 17,
    { color: C.muted });
  s.speakerNotes.textFrame.setText(notes);
  return s;
}

{
  const s = pres.slides.add();
  slideList.push(s);
  s.background.fill = C.navy;
  text(s, "药物重定位 Agent", 76, 190, 1120, 100, 64,
    { bold: true, color: C.white });
  text(s, "公开基准与肺腺癌研究案例", 80, 310, 1100, 72, 37,
    { color: C.white });
  text(s, "课程设计答辩草稿  ·  2026 年 9 月", 82, 585, 1040, 42, 22,
    { color: "#BBD5D3" });
  s.speakerNotes.textFrame.setText("内容基于仓库计划、已保存的基准结果和 LUAD 案例报告。课程任务 PDF 尚未提供，最终版式和内容须按任务要求调整。研究用途，不构成临床建议。");
}

{
  const s = slide("研究问题", 2,
    "计划来源：药物重定位Agent项目计划.md。表达反转和临床疗效的区别见 docs/limitations.md。");
  text(s, "表达反转能否改善公开基准上的药物排序？", 78, 180, 1110, 86, 34,
    { bold: true, color: C.teal });
  text(s, "肺腺癌候选能否从输入数据追踪到身份与文献证据？", 78, 318, 1110, 100, 34,
    { bold: true, color: C.navy });
  text(s, "评价重点：排名表现、数据溯源、证据边界", 80, 526, 1090, 55, 24,
    { color: C.muted });
}

{
  const s = slide("数据与质量控制", 3,
    "GEO GSE32863: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32863 ; ExperimentHub EH3226: https://bioconductor.org/packages/release/data/experiment/html/signatureSearchData.html ; 处理和哈希见 data/manifests/gse32863.json 及 eh3226-luad.json。");
  text(s, "57 对", 76, 178, 480, 100, 58, { bold: true, color: C.teal });
  text(s, "GSE32863 肿瘤与邻近正常组织", 78, 282, 500, 66, 25);
  text(s, "2 份患者编号无法配对的样本已排除", 78, 350, 500, 60, 21,
    { color: C.muted });
  text(s, "4,920", 682, 178, 500, 100, 58, { bold: true, color: C.teal });
  text(s, "A549 药名扰动签名", 682, 282, 495, 66, 25);
  text(s, "961 个与 LUAD 重合的 landmark 基因", 682, 350, 500, 70, 21,
    { color: C.muted });
  text(s, "配对 limma 与 t 检验得到相同阈值基因集：上调 512，下调 749", 78, 520,
    1110, 80, 26, { bold: true, color: C.navy });
}

{
  const s = slide("固定的分析方法", 4,
    "方法与冻结规则：docs/evaluation_protocol.md、docs/luad_screening_report.md。limma 用户指南：https://bioconductor.org/packages/release/bioc/vignettes/limma/inst/doc/usersguide.pdf。");
  text(s, "1  配对差异表达", 78, 163, 520, 54, 31,
    { bold: true, color: C.teal });
  text(s, "肿瘤减正常；FDR < 0.05 且 |log2FC| ≥ 1", 80, 220, 1100, 58, 24);
  text(s, "2  药物表达反转", 78, 316, 520, 54, 31,
    { bold: true, color: C.teal });
  text(s, "负 Spearman 与上调/下调基因秩连接分数", 80, 372, 1100, 58, 24);
  text(s, "3  排名融合", 78, 468, 520, 54, 31,
    { bold: true, color: C.teal });
  text(s, "RRF k = 60；并列分数共享名次；不读取参考治疗标签", 80, 525,
    1110, 62, 24);
}

{
  const s = slide("公开基准结果", 5,
    "数据：TRANSCRIPT v2.0.0，https://zenodo.org/records/7982976 ; 拆分和全局 AUC：stanscofi 2.0.1；三次种子 1234/1235/1236；具体均值、SD 和 NDCG 见 docs/benchmark_results.md。未知 0 按官方全局 AUC 惯例计入非正类，不是已证实临床失败。直接基线为 benchscofi 默认参数，未做发表方案的嵌套调参。");
  const chart = s.charts.add("bar", {
    position: { left: 100, top: 155, width: 1080, height: 400 },
    categories: ["B1 反转", "B0p 流行度", "B2 融合", "LogisticMF"],
    series: [
      { name: "随机拆分", values: [48.20, 73.46, 72.70, 80.05], fill: C.blue },
      { name: "弱相关拆分", values: [52.38, 50.00, 47.52, 66.96], fill: C.teal },
    ],
    barOptions: { direction: "column", grouping: "clustered" },
    hasLegend: true,
    yAxis: { min: 0, max: 100, majorUnit: 20, numberFormatCode: "0" },
    dataLabels: { showValue: true, position: "outEnd" },
  });
  applyPresentationChartFont(chart, { fontFamily: font });
  text(s, "全局 AUC × 100，三次种子的均值", 100, 580, 1050, 38, 20,
    { color: C.muted });
}

{
  const s = slide("LUAD 候选与参考药", 6,
    "完整 4,920 名排序和对照结果由 scripts/rank_luad_eh3226.py 生成；方法与限制见 docs/luad_screening_report.md。参考药名在排序前冻结于 configs/luad_positive_controls.csv。");
  text(s, "前十名集中出现类固醇相关药物", 76, 168, 1110, 70, 33,
    { bold: true, color: C.teal });
  text(s, "参考药物在 4,920 个药名中的位置", 78, 272, 1100, 58, 27,
    { bold: true });
  text(s, "docetaxel  106       crizotinib  241", 82, 341, 1110, 52, 26);
  text(s, "gefitinib  2,444     paclitaxel  2,962     erlotinib  3,147", 82,
    408, 1110, 56, 25);
  text(s, "另有 6 个预先指定药名不在该表达子集中", 80, 520, 1090, 64, 24,
    { color: C.muted });
}

{
  const s = slide("候选身份与证据缺口", 7,
    "Broad Drug Repurposing Hub 2025-08-18 注释：https://repo-hub.broadinstitute.org/repurposing ; 核对明细见 docs/luad_screening_report.md。EH3226 仅存药名，图中精确匹配是推断 GEO ID 与 Hub 样品的核对，仍不能证明上游使用了该具体样品。");
  const chart = s.charts.add("bar", {
    position: { left: 100, top: 160, width: 1080, height: 385 },
    categories: ["精确 InChIKey", "立体差异", "身份不符", "同名多 ID", "无 Hub 样品"],
    series: [{ name: "前十名数量", values: [2, 2, 1, 2, 3], fill: C.teal }],
    barOptions: { direction: "column", grouping: "clustered" },
    hasLegend: false,
    dataLabels: { showValue: true, position: "outEnd" },
  });
  applyPresentationChartFont(chart, { fontFamily: font });
  text(s, "十个候选仍全部处于“证据不足”", 100, 574, 1060, 60, 28,
    { bold: true, color: C.red });
}

{
  const s = slide("可审计的案例输出", 8,
    "代码：src/drug_repurposing_agent/luad_case.py、scripts/build_luad_case.py；TypeSafe 官方 API：https://api.typesafe.ai/docs 。目前默认运行没有 Jev 凭据，外部模型调用数为零。详见 docs/system_card.md。");
  text(s, "输入哈希与样本 QC", 76, 167, 1070, 50, 30,
    { bold: true, color: C.teal });
  text(s, "连续排名与冻结参考药清单", 76, 253, 1070, 50, 30,
    { bold: true, color: C.navy });
  text(s, "身份核对与逐候选证据账本", 76, 339, 1070, 50, 30,
    { bold: true, color: C.teal });
  text(s, "JSON 报告、执行轨迹及模型调用成本", 76, 425, 1100, 50, 30,
    { bold: true, color: C.navy });
  text(s, "默认人工复核；Jev 接口只作可选研究任务路由", 78, 552, 1100, 56, 22,
    { color: C.muted });
}

{
  const s = slide("结论与待完成工作", 9,
    "基准结果：docs/benchmark_results.md；LUAD：docs/luad_screening_report.md；进度：docs/project_status.md。课程 PDF 尚缺，无法按其具体要求确认报告、幻灯片和视频的最终格式。任何候选都不构成临床治疗建议。");
  text(s, "表达反转未超过强基线", 78, 175, 1100, 60, 34,
    { bold: true, color: C.red });
  text(s, "LUAD 前十名是待验证的研究假设", 78, 272, 1100, 70, 32,
    { bold: true, color: C.navy });
  text(s, "下一步：化合物身份核对与完整证据审阅", 78, 400, 1100, 66, 26);
  text(s, "仍需课程任务 PDF、完整内部 Eval 和 Jev 对比实验", 78, 498, 1100, 68, 24,
    { color: C.muted });
}

if (slideList.length !== 9) throw new Error("Expected nine slides");
await (await PresentationFile.exportPptx(pres)).save(candidatePath);
for (let i = 0; i < slideList.length; i++) {
  const png = await pres.export({ slide: slideList[i], format: "png", scale: 1 });
  await fs.writeFile(path.join(buildDir, `slide-${i+1}.png`),
                     new Uint8Array(await png.arrayBuffer()));
}
const result = await finalizePresentation({
  workspaceDir, candidatePath, finalPath,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000",
               "--validate-bullet-geometry", "--validate-heading-fit"],
  requiredNativeChartOwnerSlides: [5, 7],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: "design", families: [font], scriptFonts: { ea: font } },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "presentation-v2.validation.json"),
});
console.log(JSON.stringify({ finalPath, result }, null, 2));
