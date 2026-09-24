# 项目交付核验

对照项目计划第 19 节的 18 项交付物。这里的“已有”只表示对应文件或运行结果存在，不代替 PDF 课程要求和科研有效性审查。

| 计划项 | 当前证据 | 状态 |
| --- | --- | --- |
| 1 可运行药物重定位 Agent | `src/drug_repurposing_agent/`、严格模式 CLI、LUAD 案例打包器 | 部分：没有通用 LLM Planner，开放研究模式仍主要靠确定性脚本 |
| 2 RECeSS/TRANSCRIPT 适配器 | `benchmarks/recess_adapter/`、12 份三种子结果 JSON | 已有 |
| 3 外部 Benchmark 成绩表 | `docs/benchmark_results.md` | 部分：官方拆分与指标已跑，未做发表方案的嵌套调参 |
| 4 内部 Eval Suite | `tests/` 当前 19 项测试 | 部分：未达到计划中分类和数量门槛 |
| 5 Jev Choice、Score、Noul 接入层 | `src/drug_repurposing_agent/jev.py` | 部分：协议与本地模拟测试通过，缺少真实凭据与调用验证 |
| 6 Jev 置信度门控和降级 | `gate_choice` 与测试 | 部分：规则与失败回退已实现，阈值未用真实数据校准 |
| 7 Jev 与规则、通用 LLM 对比 | 无冻结决策集上的真实对比结果 | 未完成 |
| 8 LUAD 端到端案例 | `artifacts/reports/luad_case/case_report.json` | 部分：完成表达筛选与溯源，缺临床证据层完整审阅 |
| 9 Top-10 候选药物证据报告 | `docs/luad_screening_report.md` | 部分：这是筛选报告，所有候选仍为证据不足 |
| 10 Evidence Ledger | `artifacts/reports/luad_eh3226/evidence_ledger/` | 部分：十份账本存在，身份与支持/反对证据不完整 |
| 11 完整 Agent Trace 和成本报告 | `case_report.json` 的 trace/cost | 部分：覆盖当前确定性案例，不覆盖未实现的 LLM/Jev 决策 |
| 12 消融实验 | B0、B0p、B1、B1k、B2 及三个公开算法比较 | 部分：尚缺 Agent/Jev 层消融 |
| 13 受约束 RSI 演示 | 无 | 未完成 |
| 14 Docker 或锁定环境 | `requirements-benchmark.lock`、`scripts/setup_benchmark.ps1`；全新 Python 3.10 环境测试通过 | 已有 Python 环境；R/limma 另记版本 |
| 15 数据卡、系统卡和限制说明 | `docs/data_card.md`、`docs/system_card.md`、`docs/limitations.md` | 已有 |
| 16 课程设计报告 | `docs/课程设计报告_草稿.md`、`deliverables/药物重定位Agent_课程设计报告草稿.docx` | 草稿：四页已逐页检查，待任务 PDF 核对格式和内容 |
| 17 答辩 PPT | `deliverables/药物重定位Agent_答辩草稿_v2.pptx` | 草稿：九页已做版面检查，待任务 PDF 核对 |
| 18 演示视频 | 无 | 未完成 |

外部阻碍：工作区和仓库没有用户所说的任务 PDF；TypeSafe Jev 凭据也未提供。技术层面的未完成项包括更大的内部 Eval、化合物身份核对、支持与反对证据审阅、完整 Benchmark 调参/消融及演示视频。已完成的分析中，表达反转没有超过强基线，不能通过修改结论把未完成项视为完成。
