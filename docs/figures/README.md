# 答辩图表

fig1–fig8 由 `python scripts/make_figures.py [--published <RECeSS benchmark-results 目录>]` 重新生成，fig9–fig10 由 `python scripts/make_architecture_figures.py` 生成（PNG dpi 300 + SVG）；`make_figures.py` 每次运行在 `artifacts/traces/` 写入 trace。

**统一风格**（`scripts/figure_style.py`）：所有图共用一套配色（与答辩 PPT 一致）、同一字体与字号；颜色含义固定——肿瘤上调 = 红、下调 = 蓝；B2 = 红、B3 = 蓝、B4 = 绿，RECeSS 公开模型 = 灰；反转热图以 0.5 为中点，红 = 不反转、绿 = 强反转。图内不放结论性大标题，多面板图标 a/b，结论与注意事项写在报告和 PPT 的图注中。

- **fig1_nsauc_boxplot** — 11 个 RECeSS 公开 TRANSCRIPT 模型与本项目 B2、B3、B4 的逐种子 NS-AUC（`Lin's AUC` 行）箱线图，两种划分，按中位数排序。来源：`$TEMP/recess-benchmark-results-audit-20260925/results_TRANSCRIPT*/results_<MODEL>/results_N=100_*.csv`、`benchmark/results/recess_official_b2/`（`OUR_RESULT_DIRS` 中的 B3 目录存在时会自动加入）。要点：在同一官方流程、同 100 个种子下，B2 中位数随机划分 0.527（12 个中第 9）、弱相关划分 0.503（第 8），明显低于 BNNR 0.733 / MBiRW 0.740，只略高于 0.5。
- **fig2_component_ablation** — B0p/B1k/B1/B2 的 NS-AUC 均值 ± SD。来源：`benchmark/results/recess_official_component_ablation.json`。要点：RRF 融合在随机划分上优于单独 B1（0.522 vs 0.482），在弱相关划分上反而更差（0.502 vs 0.542）；B0p=0.5 是行内全部并列导致的退化值，B1k 偏低主要来自严格 “>” 下并列计 0。
- **fig3_luad_volcano** — GSE32863 57 对 LUAD 配对差异表达火山图（FDR<0.05 且 |log2FC|≥1）。来源：`data/raw/GSE32863/`，按 `scripts/process_gse32863.py` 的逻辑在内存中重算（输出 SHA-256 与 `data/manifests/gse32863.json` 一致）。要点：19,404 个基因中上调 512、下调 749，与文档一致；AGER、CLDN18、SFTPA1 等肺泡标志基因下调，SPP1、MMP11、COL1A1 等上调，符合 LUAD 生物学特征。
- **fig4_luad_top10** — Top-10 候选的身份核查分层、记录靶点数、候选特异 PMID 数和证据结论。来源：`configs/luad_top10_evidence_v1.json`（如存在 `artifacts/reports/luad_eh3226/all_candidates.csv`，则改画 RRF 分数条形图）。要点：10 个候选均恢复出精确来源 ID，但只有 2 个与 Hub InChIKey 完全一致，10/10 都是“证据不足”，只能作为研究假设。
- **fig5_top10_network** — 药物–靶点–通路三列网络，只使用证据配置中已记录的靶点和通路。来源：`configs/luad_top10_evidence_v1.json`。要点：8/10 个候选靶向 NR3C1 并记录 GR 转录通路，Top-10 其实是同一个糖皮质激素机制轴，不是 10 个互相独立的假设。
- **fig6_data_overview** — TRANSCRIPT 613×151 关联矩阵中已知 ±1 标签的散点图，以及每种疾病阳性数的直方图。来源：`data/raw/TRANSCRIPT_dataset_v2.0.0/ratings_mat.csv`。要点：已知标签只占 0.45%（401 个阳性、11 个阴性）；中位数每种疾病只有 1 个阳性，37 种疾病没有阳性，所以行式 NS-AUC 波动大、容易出现并列。
- **fig7_luad_pathways** — LUAD 疾病签名在 MSigDB Hallmark 上的富集（a 上调、b 下调，FDR < 0.05 的前 10 条）。来源：`benchmark/results/luad_pathway_enrichment_v1.json`。要点：上调侧为增殖、糖酵解、mTORC1；下调侧以 TNF-α/NF-κB 与炎症为主，更可能反映正常肺组织免疫/间质成分在肿瘤中减少。
- **fig8_top10_pathway_reversal** — Top-10 候选在各显著通路上的反转百分位热图。来源：`benchmark/results/luad_pathway_reversal_v1.json`。要点：反转集中在增殖与糖酵解；diflorasone、beclomethasone 不反转 EMT 基因。Top-10 与通路基因来自同一签名，不是独立验证。

- **fig9_architecture** — 系统架构（Agent v2）：交互层（自然语言请求 → Planner 多步规划 → 计划校验 → 决策门控）、执行状态机与“失败/计划不合法 → 观察 → 重新规划（≤3 轮）”回路、工具层（LUAD 工具链、review_literature 多 Agent、rank_transcriptome 与 manual_review）、确定性 Validator 与隔离的外部评测。组件与状态名取自 `src/drug_repurposing_agent/agent.py` 与 `agent_v2.py`。要点：LLM 只负责理解与编排，数值计算交给确定性工具，代码做最终校验。
- **fig10_workflow** — 肺腺癌端到端流程图（18 步；计划校验失败或任一步失败回到第 2 步重新规划，安全停止、引用不合格、低置信度转人工等分支），数字取自已提交的结果文件。要点：每个可能出错的环节都有明确的停止或转人工出口。
