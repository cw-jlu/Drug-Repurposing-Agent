# 课程要求核对与项目映射

核对来源：工作区文件 `生物医学信息学课程设计(1).pdf`，20 页，377,070 字节，SHA-256 `cfae1a8e3d31d26401e01908a8439dc67bc7e8660b133784a6270f6ab53146ad`。该 PDF 不随项目仓库发布。

## 明确要求

| 课程 PDF 要求 | 项目对应证据 | 当前状态 |
| --- | --- | --- |
| 真实生物医学问题 | 药物重定位与 LUAD 案例，见课程报告第 1 节 | 已覆盖 |
| Problem → Data → Model → Benchmark → Biological interpretation | 报告第 1–6 节、数据卡、Benchmark 结果和 LUAD 筛选报告 | 已覆盖，PPT 仍需明确标注五环节 |
| GitHub 代码 | 可安装 Python 包、CLI、脚本、测试、数据 Manifest | 已覆盖 |
| 约 10 分钟汇报 | 九页可编辑答辩稿 v6 与讲述提纲 | 结构匹配，已重新渲染；仍需本人实际计时排练 |
| 自然语言问题输入 | CLI `--question` 与 `run_agent_task` | 已实现 |
| Agent 自动规划任务 | `RulePlanner`、`StructuredPlanner`、DeepSeek 与经验证的 `AgentPlan` | 已完成真实 DeepSeek 调用；冻结 v3 规划/工具选择为 98/100，不是药效评测 |
| LLM + tool calling | provider-neutral 工具 Schema、DeepSeek 真实函数调用与本地权限校验 | 已实测；15 条方法选择响应经 provider-visible trace 核验，效果仍低于固定 B2 |
| 最终是可自动完成分析的智能体系统，而非单一模型 | Agent 路由、确定性工具、证据门控、Trace | 核心链路已具备；开放证据审阅仍需人工完成 |

## PDF 没有规定的事项

课程 PDF 没有写明必须提交单独书面报告、演示视频，也没有规定封面、页数、引用格式或文件命名。项目可保留报告和视频作为增强交付，但不得把这些自定项误写成教师硬性要求。

## 汇报重点

课程 PDF 明确强调不只看 accuracy，可分别展示问题价值、技术设计、实验严谨性、解释性和展示效果。因此答辩应保留“表达反转未超过强基线”这一负结果，并用标签隔离、数据哈希、错误停机、工具白名单和候选证据不足门控说明系统价值，不应包装成临床疗效或 SOTA。
