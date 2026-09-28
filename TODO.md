# 待办事项（2026-09-28 更新）

当前代码、基准、LUAD 表达筛选和交付草稿的状态见 [`docs/completion_audit.md`](docs/completion_audit.md)。本清单供下次继续；勾选前须核对实际文件和运行结果。

## 已核对

- [x] 已取得并逐页核对课程任务 PDF；要求映射见 `docs/assignment_alignment.md`。PDF 只要求五环节、GitHub 代码和约 10 分钟汇报，没有规定单独报告、视频、封面或页数格式。
- [x] 已重新运行 `python -m pytest -q`；当前环境为 46 项通过。
- [x] 已补充自然语言 Agent 路由、工具 Schema、Strict/Open 权限、缺输入/失败停机和完整执行轨迹；外部结构化 LLM 通过 `StructuredPlanner` 接入，默认使用可复现的规则降级。
- [x] 已按课程要求刷新 Word 报告 `deliverables/archive/药物重定位Agent_课程设计报告_v2.docx` 与九页答辩稿 `deliverables/archive/药物重定位Agent_答辩稿_v4.pptx`；DOCX 四页和 PPTX 九页均完成逐页视觉检查，PPTX 的两张图表保持原生可编辑。
- [x] 已接入 DeepSeek `deepseek-flash` 真实函数调用；冻结 20 个规划案例后运行三次，准确率为 0.95、0.85、1.00，均值 0.933。真实 TRANSCRIPT 613×151 全流程已跑通。

## 2026-09-28 新增：提分与展示（已暂停，见各项状态）

背景：B2 在官方 100 次 NS-AUC 中排 9/12 与 8/12。复核发现官方 NS-AUC（Lin's AUC）按**药物行**对疾病排序，并用严格 `>` 将并列计为 0；B2 按疾病列排序、且流行度分量在行内为常数，方向与评价指标不一致。

- [x] **B3 行方向无训练融合**：疾病流行度、药物/疾病表达 kNN、标签共现 CF、反转，按行归一化排名后平均，并加入并列打破。配置已在正式评分前冻结于 `configs/b3_row_fusion_v1.json`，开发种子与官方 100 个种子不相交。弱相关拆分只有一个固定留出集，冻结前已看过，已在配置中披露；B3noREV 标记为事后变体。
- [x] B3 官方 Runner 100 次 × 两种拆分已完成：随机 0.7234（13 个中排第 2，BNNR 0.7331 第 1），弱相关 0.6919（排第 3，MBiRW 0.7384 第 1）。见 `docs/recess_official_comparison.md`。
- [ ] B3noREV（事后变体）暂停于随机拆分 46/100；重新运行 `scripts/run_recess_official_b3.py --models B3noREV` 会复用缓存继续。
- [ ] **第二轮 B4 = B3 + BNNR（NumPy 移植）**：已暂停。`benchmarks/recess_adapter/bnnr_numpy.py` 为未验证草稿（未提交）。步骤：在 3 个官方种子上核对移植保真度 → 只在开发种子上设计集成 → 冻结提交 → 官方 100 次。
- [ ] **LLM 知识污染探针**（已暂停，草稿未提交）：开卷（给药名/病名）对比闭卷（只给表达特征），论证 Strict Mode 的必要性（`docs/contamination_probe.md`）。
- [ ] **多 Agent 证据审阅**（已暂停，草稿未提交）：LUAD Top-10 由 Literature Agent、Critic Agent、确定性引用校验、Coordinator 分级完成，统计幻觉引用拒绝率（`docs/multi_agent_review.md`）。
- [ ] **决策层消融（Jev 替代）**（已暂停，草稿未提交）：冻结 ≥100 个 Choice/Score/Noul 决策用例，比较 J0 规则、J1 通用 LLM、J3 LLM+置信度门控；Jev 保留接口，获得权限后在同一冻结集复测（`docs/decision_eval_v1.md`）。
- [x] **交互式 Demo**：Streamlit，含 Agent 运行/回放、LUAD 候选证据卡、Benchmark 表（`app/demo.py`、`docs/demo.md`）。
- [x] **答辩图表**：12 模型 NS-AUC 箱线图、组件消融、LUAD 火山图、Top-10 条形图与药物—靶点—通路网络、数据稀疏度（`scripts/make_figures.py` → `docs/figures/`）。
- [ ] 将上述结果写入报告 v5 / 答辩稿 v7，逐页核验；旧版交付物归档到 `deliverables/archive/`。
- [ ] 按 `docs/defense_rehearsal_10min.md` 计时排练一次。
- [ ] **轮换 DeepSeek API Key**：旧 key 曾在对话中暴露，需本人在 DeepSeek 控制台操作。
- [ ] 推送 `codex/deepseek-planner` 并开 PR 合并到 `main`，让 GitHub 展示最新版本。

## 恢复工作时先处理

- [ ] 确认是否提供 TypeSafe Jev 的可用访问凭据。若无法取得，保留接口与模拟测试，但明确标注 Jev 实测及对比未完成。决策层对比先用 J0/J1/J3 完成（见上）。

## 科研与系统工作

- [x] 已扩展并冻结 61 项 Planner Eval v2，覆盖十类任务与安全边界；保存修改前与加固后结果。
- [x] 已在任何 v3 调用前冻结独立的 100 项 Agent v3 holdout 及其 SHA-256；规则 Planner 为 90/100，未按 v3 调参的 DeepSeek Planner 为 98/100。
- [x] 已对 B1k/B2 在三次外层种子和两种拆分上完成三折内层嵌套调参，并保存泄漏审计；随机拆分有改善，弱相关拆分没有改善。
- [x] 已对 ALSWR、PMF、LogisticMF 在两种拆分、五个外层种子和三折内层交叉验证上完成嵌套调参；十次运行的外层泄漏审计均通过。
- [x] 已将 B2 接入固定版本的 RECeSS 官方 Runner，在两种拆分下与作者公布的 11 模型逐种子对齐 100 次、五折选模和 NS-AUC；B2 分别排 9/12 与 8/12。官方五折没有参数网格搜索，弱相关拆分重复同一个外层留出集。
- [x] 已在同一官方 Runner 下拆开 B0p、B1k、B1 并完成两种拆分各 100 次消融；已在成绩揭晓前冻结一次 DeepSeek 方法选择，并对 LUAD Top-10 完成一次受约束的研究行动审阅。两次选择与审阅都只是单次模型样本，不是药效或泛化证明。
- [x] 已对 LUAD Top-10 逐一整理身份状态、名称级靶点/通路、候选特异和类别级支持/反对文献，并重建 10/10 源 GSE92742 `sig_id` 与 `pert_id`。
- [x] 已冻结剂量、LUAD 分子亚型、NR3C1 机制和 6×6 联合用药的实验方案及测量模板；这些是预注册方案，不是已产生的湿实验结果。
- [ ] 若取得 Jev 凭据，在冻结决策集上实测 Choice、Score、Noul，校准置信度门控，并与规则和通用 LLM 比较准确率、校准、延迟及成本。
- [ ] 补齐 Jev 层消融与受约束 RSI 演示；DeepSeek 与规则 Planner 的初步对比、失败回退和调用成本已经记录。

## 课程交付

- [ ] 若用于最新答辩，将新增的 RECeSS 100 次 NS-AUC 对比、B2 消融和 LLM 决策边界写入 Word/PPT 新版本，再逐页核验；现有 v2/v4 是这些对比前的版本。
- [ ] 如教师另行要求，再录制演示视频；当前课程 PDF 只明确约 10 分钟汇报，没有要求视频或单独书面报告。
- [x] 已对照项目计划第 19 节逐项复核 `docs/completion_audit.md`；只将有直接证据的项目标为完成。

当前关键限制：纯表达反转在现有公开基准上没有优于强基线（它是负结果，不是提分来源）；任何提分都来自已知关联结构。LUAD 候选均不能作为临床用药建议。
