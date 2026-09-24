# 项目交付核验

对照项目计划第 19 节的 18 项交付物。这里的“已有”只表示对应文件或运行结果存在，不代替课程要求和科研有效性审查。课程 PDF 的实际要求映射见 `assignment_alignment.md`。

| 计划项 | 当前证据 | 状态 |
| --- | --- | --- |
| 1 可运行药物重定位 Agent | `agent.py`、DeepSeek Planner、自然语言 CLI、严格模式表达工作流、LUAD 案例打包器 | 已有核心链路：真实 DeepSeek 规划已驱动完整 TRANSCRIPT 分析；开放证据审阅仍未自动化 |
| 2 RECeSS/TRANSCRIPT 适配器 | `benchmarks/recess_adapter/`、12 份三种子结果 JSON | 已有 |
| 3 外部 Benchmark 成绩表 | `docs/benchmark_results.md` | 部分：B1k/B2 已完成两种拆分、三次外层种子和三折内层调参；benchscofi 三模型仍为默认参数 |
| 4 内部 Eval Suite | `tests/` 当前 32 项测试，另有冻结的 20 项 v1 和 61 项 v2 Planner Eval | 已有较完整路由回归；v2 加固后结果不是独立 holdout，仍需新增盲测集 |
| 5 Jev Choice、Score、Noul 接入层 | `src/drug_repurposing_agent/jev.py` | 部分：协议与本地模拟测试通过，缺少真实凭据与调用验证 |
| 6 Jev 置信度门控和降级 | `gate_choice` 与测试 | 部分：规则与失败回退已实现，阈值未用真实数据校准 |
| 7 Jev 与规则、通用 LLM 对比 | `planner_eval_results.md` 已完成规则与 DeepSeek 对比 | 部分：通用 LLM 对比已有，Jev 仍无真实凭据和结果 |
| 8 LUAD 端到端案例 | `artifacts/reports/luad_case/case_report.json`、`docs/luad_top10_evidence_matrix.md` | 部分：完成表达筛选、溯源和文献/靶点分诊，缺化合物确认与实验验证 |
| 9 Top-10 候选药物证据报告 | `docs/luad_screening_report.md`、`docs/luad_top10_evidence_matrix.md` | 已完成候选级核查；结论仍是全部证据不足，不构成疗效报告 |
| 10 Evidence Ledger | `artifacts/reports/luad_eh3226/evidence_ledger/` | 部分：十份账本存在，身份与支持/反对证据不完整 |
| 11 完整 Agent Trace 和成本报告 | `agent_run.json`、Planner Eval JSON 与 `case_report.json` | 部分：已覆盖 DeepSeek 规划和确定性工具执行；不覆盖尚未实测的 Jev 调用 |
| 12 消融实验 | B0、B0p、B1、B1k、B2 及三个公开算法比较 | 部分：尚缺 Agent/Jev 层消融 |
| 13 受约束 RSI 演示 | 无 | 未完成 |
| 14 Docker 或锁定环境 | `requirements-benchmark.lock`、`scripts/setup_benchmark.ps1`；全新 Python 3.10 环境测试通过 | 已有 Python 环境；R/limma 另记版本 |
| 15 数据卡、系统卡和限制说明 | `docs/data_card.md`、`docs/system_card.md`、`docs/limitations.md` | 已有 |
| 16 课程设计报告 | `docs/课程设计报告_草稿.md`、`deliverables/药物重定位Agent_课程设计报告草稿.docx` | 已刷新：包含最新 Agent 结果，四页已逐页检查；课程 PDF 未规定必须提交单独报告 |
| 17 答辩 PPT | `deliverables/药物重定位Agent_答辩稿_v3.pptx` | 已刷新：九页结构符合约 10 分钟汇报，包含最新 Agent Trace，两张图表保持原生可编辑并已逐页检查 |
| 18 演示视频 | 无 | 未完成 |

外部阻碍：TypeSafe Jev 凭据尚未提供。课程 PDF 已在工作区找到并完成核对，DeepSeek 通用 LLM 已完成真实实测。技术层面的未完成项包括更大的内部 Eval、化合物身份核对、支持与反对证据审阅、完整 Benchmark 调参/消融，以及按需制作的演示材料。已完成的分析中，表达反转没有超过强基线，不能通过修改结论把未完成项视为完成。
