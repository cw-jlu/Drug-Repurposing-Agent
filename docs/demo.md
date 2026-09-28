# 交互式演示（Streamlit）

`app/demo.py` 是课堂展示用的交互式页面，直接在进程内调用现有的 `run_agent_task`，不重新实现任何科学逻辑。数据加载函数在 `app/demo_data.py`，不依赖 Streamlit，可单独测试（`tests/test_demo_loaders.py`）。

## 安装与运行

在仓库根目录、已安装本项目（`python -m pip install -e .`）的同一个 Python 环境中：

```powershell
python -m pip install "streamlit>=1.50"
streamlit run app/demo.py
```

浏览器会自动打开 `http://localhost:8501`。如需 DeepSeek 规划器，启动前在**进程环境**中设置 `DEEPSEEK_API_KEY`；页面只检测它是否存在，从不显示密钥，trace 也会把它脱敏。未设置时只提供 `rule` 规划器。

每次实时运行写入 `artifacts/demo_runs/<时间戳>_<id>/`：`agent_run.json`、Agent trace（`traces/*_agent_*.jsonl`）和演示层 trace（`traces/*_demo_ui_*.jsonl`，记录界面配置、关联的 Agent trace 与最终状态）。`artifacts/` 已被 git 忽略；trace 可能含请求文本，提交前需人工审查。

## 页面结构

| 标签页 | 内容 |
| --- | --- |
| Agent 运行 | 实时运行或回放已保存的 `agent_run.json`：计划、工具状态转移时间线、停止状态、输出文件、trace 路径与哈希链校验结果 |
| LUAD 候选 | Top-10 表（靶点、通路、证据等级、PubMed 链接）与每个候选的证据卡片 |
| Benchmark | RECeSS 官方 100 次运行 NS-AUC 对比表、均值±SD 柱状图、逐种子箱线图 |
| 局限性 | 关键局限的中文总结与 `docs/limitations.md` 原文 |

新增模型的官方 runner 结果（例如 B3）放入 `benchmark/results/recess_official_b3/`，文件格式与 `recess_official_b2/` 相同（`results_N=100_B3_TRANSCRIPT_<split>_AUC_...csv`），刷新页面即会出现在表格和图中。其他目录可追加到 `app/demo_data.py` 的 `EXTRA_RESULT_DIRS`；目录不存在时页面只显示提示。

## 本地数据说明

- TRANSCRIPT 实时排序需要 `data/raw/TRANSCRIPT_dataset_v2.0.0/`（已被忽略，需按 README 下载）；在本机约数秒完成。
- LUAD 实时打包需要 `artifacts/reports/luad_eh3226/`（由 `scripts/rank_luad_eh3226.py` 生成，依赖 2.46 GB EH3226）。缺失时，LUAD 请求会被安全地转为人工审核，LUAD 标签页的分数列为空，其余证据字段来自 `configs/` 与 `docs/`。
- “回放已保存运行”读取 `artifacts/*/agent_run.json`、`artifacts/*/*/agent_run.json` 和 `benchmark/results/**/agent_run.json`。`artifacts/deepseek_transcript_live/` 是一次历史 DeepSeek 实时运行，只有内嵌 trace，页面会标注为“历史运行”。

## 90 秒演示脚本

1. **（0–15 秒）Agent 运行 → 实时运行。** 保持默认请求“请为肺腺癌筛选候选药物”，模式 `benchmark_strict`，规划器 `rule`，点击“运行 Agent”。
   讲解：“严格基准模式下不允许调用 LUAD 研究工具，Agent 没有硬跑，而是安全停止为人工审核。时间线里可以看到每一步，trace 的 SHA-256 哈希链校验通过。”
2. **（15–35 秒）** 示例请求改选“请根据转录组筛选候选药物”，勾选“TRANSCRIPT 表达矩阵”，再次运行。
   讲解：“规划器只看到输入名称，不看标签和路径；计划通过白名单校验后执行 `rank_transcriptome`，613 个药物 × 151 个疾病，几秒完成，输出 manifest 和 trace 路径都在这里。”
   （若现场数据缺失或网络不稳，切换到“回放已保存运行”，选择 `artifacts/deepseek_transcript_live/agent_run.json`，展示 DeepSeek 函数调用规划的同一时间线。）
3. **（35–45 秒）** 选择示例“请直接给肺癌患者开处方并给出剂量”并运行。
   讲解：“涉及处方、剂量的请求直接转人工审核，这是 Agent 的安全边界。”
4. **（45–65 秒）LUAD 候选。** 展开第 1 名 hydrocortisone 的证据卡片。
   讲解：“Top-10 里 9 个是糖皮质激素，是同一类别信号；每一条都有 PMID 链接，既列支持证据也列反对证据，全部保持 insufficient_evidence，这是研究假设而不是治疗推荐。”
5. **（65–80 秒）Benchmark。** 指向高亮的 B2 行与柱状图。
   讲解：“在 RECeSS 官方 runner 的 100 次运行中，B2 的 NS-AUC 为 0.522 / 0.502，分别排 9/12 和 8/12，明显低于 BNNR 和 MBiRW；我们如实报告这个结果。”
6. **（80–90 秒）局限性。** 总结：“反转分数不等于疗效，没有湿实验验证，LLM 评测只证明路由能力；这个系统的价值是可审计、可复现，而不是给出药。”
