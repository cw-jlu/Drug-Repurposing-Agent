# 项目交付核验

对照项目计划第 19 节的 18 项交付物。这里的“已有”只表示对应文件或运行结果存在，不代替课程要求和科研有效性审查。课程 PDF 的实际要求映射见 `assignment_alignment.md`。

| 计划项 | 当前证据 | 状态 |
| --- | --- | --- |
| 1 可运行药物重定位 Agent | `agent.py`、DeepSeek Planner、自然语言 CLI、严格模式表达工作流、LUAD 案例打包器 | 已有核心链路：真实 DeepSeek 规划已驱动完整 TRANSCRIPT 分析；开放证据审阅仍未自动化 |
| 2 RECeSS/TRANSCRIPT 适配器 | `benchmarks/recess_adapter/`、12 份三种子结果 JSON、官方 Runner B2 与三分支补丁及对应 100 次结果 CSV | 已有 |
| 3 外部 Benchmark 成绩表 | `docs/benchmark_results.md`、`benchmark/results/recess_official_b2_vs_11.json`、`benchmark/results/recess_official_component_ablation.json` | 已有：B2 与作者 11 模型在相同 100 种子、五折选模、NS-AUC 下对比；B0p/B1k/B1 消融完成；单次 LLM 预选与研究行动审阅单列，不作为药效准确率 |
| 4 内部 Eval Suite | `tests/`、冻结的 20 项 v1、61 项 v2 和独立 100 项 v3 Planner Eval；另有单例 LUAD 研究分诊审计 | 已有：旧 v3 规则为 90/100、DeepSeek 为 98/100；新 trace 评分是回归检查，不是第二个独立 holdout。15/15 provider-visible 方法选择 trace 核验通过，但事后正例/未知修订实验中选法不及固定 B2（−0.11031 NS-AUC）。LUAD 单例审计不是独立 Agent holdout |
| 5 Jev Choice、Score、Noul 接入层 | `src/drug_repurposing_agent/jev.py` | 部分：协议与本地模拟测试通过，缺少真实凭据与调用验证 |
| 6 Jev 置信度门控和降级 | `gate_choice` 与测试 | 部分：规则与失败回退已实现，阈值未用真实数据校准 |
| 7 Jev 与规则、通用 LLM 对比（2026-10-01 已实测，见 `docs/jev_evaluation.md`） | `planner_eval_results.md` 已完成规则与 DeepSeek 对比 | 部分：通用 LLM 对比已有，Jev 仍无真实凭据和结果 |
| 8 LUAD 端到端案例 | `artifacts/reports/luad_case/case_report.json`、`configs/luad_top10_signature_ids.csv`、`docs/luad_top10_evidence_matrix.md` | 部分：完成表达筛选、逐签名 ID 恢复、溯源和文献/靶点分诊；湿实验尚未执行 |
| 9 Top-10 候选药物证据报告 | `docs/luad_screening_report.md`、`docs/luad_top10_evidence_matrix.md` | 已完成候选级核查；结论仍是全部证据不足，不构成疗效报告 |
| 10 Evidence Ledger | `artifacts/reports/luad_eh3226/evidence_ledger/` | 部分：十份账本存在，身份与支持/反对证据不完整 |
| 11 完整 Agent Trace 和成本报告 | 新运行的 SHA-256 链式 JSONL trace、`agent_run.json`、Planner Eval JSON 与 `case_report.json`；Planner 与方法选择的 provider-visible trace 评分器 | 部分：新入口记录成功与失败，能检查记录完整性及报告、模型请求/响应、工具调用的一致性；历史 v3 DeepSeek 的原始响应不能补录，Jev 尚未实测 |
| 12 消融实验 | B0、B0p、B1、B1k、B2 及三个公开算法比较 | 部分：尚缺 Agent/Jev 层消融 |
| 13 受约束 RSI 演示 | 无 | 未完成 |
| 14 Docker 或锁定环境 | `requirements-benchmark.lock`、`scripts/setup_benchmark.ps1`；全新 Python 3.10 环境测试通过 | 已有 Python 环境；R/limma 另记版本 |
| 15 数据卡、系统卡和限制说明 | `docs/data_card.md`、`docs/system_card.md`、`docs/limitations.md` | 已有 |
| 16 课程设计报告 | `docs/课程设计报告_v5.md`、`deliverables/药物重定位Agent_课程设计报告_v5.docx`、`deliverables/latex/course_report_v5.pdf` | v5 加入 B3/B4 官方结果、污染探针、决策层消融、多 Agent 审阅与图表；文献部分写明 65/65 通过 PMID/逐字校验，但 23 条来源/药物指向经事后裁定后 13 条不再作为候选级证据，10/10 证据不足保持不变；v4 已归档 |
| 17 答辩 PPT | `deliverables/药物重定位Agent_答辩稿_v7.pptx` | v7 十一页由脚本从已提交结果生成，三张原生可编辑图表；第 10、11 页已加入 23 条范围预筛及 13 条排除的边界；v6 已归档 |
| 18 演示视频 | 无 | 未完成 |

外部阻碍：TypeSafe Jev 凭据尚未提供；剂量、亚型和组合验证需要湿实验资源。课程 PDF 已完成核对，DeepSeek 通用 LLM 已真实实测，100 例 v3 独立 holdout、逐签名 ID 恢复和三种官方基线的五种子嵌套调参均已完成。表达反转没有超过强基线，不能通过修改结论把尚未执行的实验视为完成。
