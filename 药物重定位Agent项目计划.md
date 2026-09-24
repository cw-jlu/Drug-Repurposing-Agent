# 面向药物重定位的可审计生物医学 Agent：完整项目计划

> 文档状态：项目执行基线<br>
> 更新日期：2026-09-24<br>
> 项目类型：纯 Agent 系统，不训练新的机器学习预测模型<br>
> 核心评测结构：第三方公开 Benchmark + 内部 Eval Suite<br>
> Jev 含义：TypeSafe AI 发布的 System One 决策模型，不是 Judge–Evaluator–Verifier 的缩写<br>
> **⚠ 2026-09-24 预实验结论（详见第 23 节）：纯表达反转在 TRANSCRIPT 上 AUC≈0.48，与随机无差别；不能以“反转法刷 SOTA”为目标，外部 Benchmark 主线需按第 23 节调整。**

---

## 1. 项目定位

### 1.1 建议题目

**面向药物重定位的可审计生物医学 Agent：基于转录组反转、多源证据和 Jev 决策门控的端到端系统**

英文题目：

**An Auditable Biomedical Agent for Drug Repurposing via Transcriptomic Reversal, Multi-source Evidence, and Jev-based Decision Gating**

### 1.2 核心目标

开发一个不训练新预测模型、但能自主调用生物信息学工具的药物重定位 Agent。系统需要完成：

1. 理解疾病重定位任务与约束；
2. 获取并检查疾病转录组数据；
3. 构建疾病表达签名；
4. 检索或加载药物扰动表达签名；
5. 使用确定性方法计算疾病—药物反转关系；
6. 检索靶点、通路、药理、临床与文献证据；
7. 融合多个分数并生成候选药物排序；
8. 识别证据不足、证据冲突和数据质量问题；
9. 输出可追溯、可复现的证据报告；
10. 在第三方公开 Benchmark 上接受外部评测；
11. 使用内部 Eval Suite 进行开发、回归测试和版本验收；
12. 探索 Jev 决策模型在 Agent 高频、封闭式决策节点中的作用。

### 1.3 项目创新点

本项目不以“训练一个更高分的预测模型”为核心，而以“构建一个可靠的科研 Agent 系统”为核心：

- LLM 负责开放式任务理解、规划、解释和报告生成；
- 确定性生物信息学工具负责统计计算；
- Jev 负责适合 Choice、Score、Noul 的快速结构化判断；
- 普通代码负责权限、规则、数据隔离、阈值和执行控制；
- 外部 Benchmark 评价最终任务结果；
- 内部 Eval Suite 评价过程质量、可靠性、成本与版本退化。

---

## 2. 项目边界

### 2.1 本项目包含

- 单 Agent 为主的结构化工作流；
- GEO、LINCS、Open Targets、PubMed、Repurposing Hub 等工具接入；
- 差异表达、基因集富集、表达反转与规则排序；
- 第三方 Benchmark 适配器；
- 内部 Eval Suite；
- Jev 决策节点实验；
- 完整执行轨迹、数据版本和证据账本；
- 肺腺癌端到端案例；
- 后续受约束的 RSI 实验。

### 2.2 本项目不包含

- 训练或微调新的分类器、排序器、神经网络或基础模型；
- 用外部 Benchmark 测试答案修改提示词或阈值；
- 将未知关联自动当作负例；
- 将表达反转直接解释为临床有效；
- 将体外细胞系结果直接推广到患者；
- 将内部 Eval Suite 称为 Benchmark；
- 让 Jev 代替统计计算、文献生成或开放式科研推理。

### 2.3 “纯 Agent、无机器学习”的具体含义

项目系统本身不训练预测模型。允许使用：

- 已有基础模型的推理 API；
- Jev 的推理 API；
- 差异表达统计；
- Spearman 相关；
- 基因集富集；
- Connectivity Score；
- Reciprocal Rank Fusion；
- 固定规则和阈值；
- 已发布算法的公开结果作为基线；
- **无参数拟合的记忆型方法**：基于训练折已知关联的流行度（degree）、药物/疾病表达相似度 kNN 标签传播。它们只读取官方训练折标签、不做梯度或参数学习，须在报告中明确标注为“使用训练标签的无训练方法”。

如果为了复现第三方 Benchmark 而运行其官方机器学习基线，这些基线只用于比较，不属于我们系统的实现方案。

---

## 3. 研究问题

### RQ1：端到端能力

Agent 能否从疾病转录组数据出发，自动生成具有完整计算轨迹和证据链的候选药物列表？

### RQ2：Agent 架构价值

结构化状态机 Agent 是否比一次性提示、普通 ReAct Agent 更稳定、更可复现？

### RQ3：确定性工具价值

多种确定性转录组反转方法加排名融合，是否优于单一相关系数方法？

### RQ4：Jev 集成价值

在工具路由、证据分级、完成条件判断和风险门控等封闭式决策节点中，Jev 是否能在保持决策质量的同时降低延迟、成本和输出解析失败？

### RQ5：验证与改进

基于内部 Eval Suite 的受约束改进流程，能否提高系统可靠性而不导致外部 Benchmark 过拟合？

---

## 4. Benchmark 与内部 Eval 的术语规范

### 4.1 External Benchmark

只有满足以下条件的第三方资源才称为 Benchmark：

- 任务和数据由第三方定义；
- 具有固定标签、答案或参考结果；
- 具有公开或可复现的评价协议；
- 可以与其他系统进行横向比较；
- 我们不控制其最终评分规则。

用途：回答“我们的 Agent 在别人定义的任务上表现如何”。

### 4.2 Internal Eval Suite

我们为开发和验收维护的测试集合称为：

- Internal Eval Suite；
- Regression Evals；
- Acceptance Evals；
- Failure-injection Evals。

用途：回答“新版本是否更可靠、是否发生退化、是否满足上线门槛”。

### 4.3 论文和答辩中的统一表述

> 我们实现了一个药物重定位 Agent，并在第三方公开 Benchmark 上进行外部评测；同时构建内部 Eval Suite，用于工具验证、工作流回归、Jev 接入评估和系统迭代。

不能写成：

> 我们自己构建了一个 Benchmark。

---

## 5. 外部 Benchmark 方案

## 5.1 主 Benchmark：RECeSS Drug Repurposing Benchmark

RECeSS 团队已经公开药物重定位 Benchmark 代码，并在 Scientific Reports 发表配套研究。公开资源包括：

- 8 个药物重定位数据集；
- 11 种基线算法；
- 随机拆分与弱相关拆分；
- 5 折交叉验证配置；
- AUC、NDCG、F-score、Accuracy 和运行时间等指标；
- 完整复现脚本和随机种子记录。

项目主数据集使用其中的 **TRANSCRIPT v2.0.0**。

### TRANSCRIPT 已确认数据

2026-09-24 已从 Zenodo 实际下载 v2.0.0 压缩包并校验 MD5。**README 的 204 种药物、116 种疾病指至少存在一条非零已知关联的实体数；实际 CSV 特征与关联矩阵覆盖 613 种药物、151 种疾病。**完整候选空间为 613 × 151 = 92,563 个药物—疾病组合，其中 401 个正关联、11 个负关联、92,151 个未知关系。两种口径必须分别报告，不能把 204 × 116 写成完整矩阵尺寸。

- `ratings_mat.csv`：613 行药物 × 151 列疾病；
- `items.csv`：12,096 行基因 × 613 列药物；
- `users.csv`：12,096 行基因 × 151 列疾病；
- 204 种药物和 116 种疾病至少有一条正/负已知关联；
- 压缩包 31,599,556 字节，MD5 为 `67b5be71611361ca493303b052a4944c`；
- 药物表达来自 CREEDS 或 LINCS L1000，疾病表达来自 CREEDS；
- `ratings_mat.csv` 中 `1` 为已知正关联，`-1` 为记录的失败关联，`0` 为未知状态。

### 接入方法

实现 Benchmark 兼容适配器：

```text
fit(dataset)
    不训练模型
    只验证特征、ID、维度和配置

predict_proba(dataset)
    Agent 制定批量计算计划
    确定性工具生成完整药物—疾病分数矩阵
    Agent 检查异常并输出标准格式
```

Agent 不为每个药物—疾病对单独调用 LLM，而是调度批处理工具一次性计算矩阵。这样能控制成本，也更符合企业 Agent 中“模型负责决策、工具负责计算”的设计。

### Benchmark Strict Mode

允许访问：

- Benchmark 提供的药物和疾病表达特征；
- 不包含答案的数据字典；
- 确定性统计工具；
- 基因和本体 ID 映射表。

禁止访问：

- 测试标签；
- 完整关联矩阵；
- repoDB；
- 直接查询药物适应症的接口；
- “药物 X 是否治疗疾病 Y”式网络搜索；
- 临床成功或失败结果字段。

### 主评价指标

- Row-wise AUC；
- Global AUC；
- NDCG；
- F-score；
- Accuracy；
- Recall@10、Recall@20 和 MRR（项目补充指标，需公开计算定义及适用病例）；
- 数据覆盖率；
- 运行时间。

### 公平性要求

- 保留官方拆分；
- 保留随机种子；
- 不只报告最好的一次；
- 输出均值、标准差和置信区间；
- 所有版本使用相同数据与资源预算；
- 在查看正式测试成绩前冻结版本。

## 5.2 次 Benchmark：repoDB

repoDB 用于检验系统对真实获批关系和临床失败关系的排序能力。

执行前必须进行覆盖率审计：

1. 映射 repoDB 药物 ID；
2. 映射疾病本体 ID；
3. 检查疾病表达签名覆盖率；
4. 检查药物扰动表达覆盖率；
5. 记录无法映射的原因；
6. 在查看标签前冻结纳入规则。

如果只覆盖一部分，结果名称必须写成：

**repoDB-covered subset evaluation**

不能写成完整 repoDB Benchmark 成绩。

repoDB 中的失败关系不能被解释为绝对无效，因为失败可能源于毒性、剂量、招募或试验设计等原因。

## 5.3 辅助 Benchmark：BioAgent Bench

选择一个与转录组最接近的任务，用于评价：

- 数据发现；
- 数据质量检查；
- 差异表达；
- 通路分析；
- 代码执行；
- 文件生成；
- 错误恢复；
- 输出格式遵循。

它只能作为通用生物信息学 Agent 能力的辅助证据，不能替代药物重定位主 Benchmark。

## 5.4 候选扩展：RepurposingBench

RepurposingBench 的任务形式与本项目高度匹配，但当前公开页面没有提供完整测试数据和评分代码，需要联系发布方获取。

因此：

- 不列入必须完成项；
- 不承诺一定能运行；
- 如果成功获得数据和协议，再作为附加外部 Benchmark；
- 不自行猜测或重建其隐藏答案。

## 5.5 两条数据链路：输入、用途和边界

| 链路 | 主要输入 | 处理 | 输出与评价 |
|---|---|---|---|
| 公开 Benchmark | TRANSCRIPT v2.0.0 的 `items.csv`、`users.csv`；官方划分后仅训练部分可见的 `ratings_mat.csv` | 对 613 种药物和 151 种疾病的表达向量按基因对齐，批量计算反转分数 | 613×151 连续分数矩阵；由第三方协议评价被保留的关联。主实验不得查询适应症与结论性文献 |
| LUAD 科研案例 | GSE32863 的患者肿瘤/邻近正常表达、LINCS GSE92742 的 A549 化合物扰动签名 | 配对差异表达、药物反转、靶点/文献核查、证据分级 | Top-10 可追溯候选报告。它是案例展示，不冒充外部 Benchmark 成绩 |

这两条链路共用 Agent 编排与统计工具，但分开保存数据和结果。TRANSCRIPT 的预处理疾病向量不能当成 GSE32863 原始患者样本；LUAD 的开放式文献检索结果也不能倒灌到 TRANSCRIPT 的盲测排名中。

## 5.6 TRANSCRIPT：如何获取、文件长什么样、怎样使用

**原始来源**：[Zenodo v2.0.0，DOI 10.5281/zenodo.7982976](https://zenodo.org/records/7982976)。数据构建流程由 [RECeSS 数据仓库](https://github.com/RECeSS-EU-Project/drug-repurposing-datasets) 的 `TRANSCRIPT_dataset.ipynb` 说明；评测协议见 [RECeSS benchmark-code](https://github.com/RECeSS-EU-Project/benchmark-code)。

**获取与核验**：下载 `TRANSCRIPT_dataset_v2.0.0.zip`；保留 Zenodo record ID、下载日期、原始 URL、文件字节数和 MD5；确认校验值为 `67b5be71611361ca493303b052a4944c` 后解压。压缩包内除 `LICENSE`、`README` 外，恰好包含以下三个 CSV。2026-09-24 已实际读取压缩包目录、README、CSV 表头和标签计数；这不是仅根据网页推算。

| 文件 | 实测维度 | 行 ID / 列 ID | 用途 |
|---|---:|---|---|
| `ratings_mat.csv` | 613 × 151 | 行是 DrugBank ID 或 `CID...`；列是疾病 MedGen Concept ID | `1/-1/0` 关联标签；仅分割器与评分器读取完整文件 |
| `items.csv` | 12,096 × 613 | 行是 HGNC gene symbol；列是药物 ID | 药物扰动后的表达变化向量，供 Agent 排名 |
| `users.csv` | 12,096 × 151 | 行是 HGNC gene symbol；列是疾病 MedGen ID | 疾病表达变化向量，供 Agent 排名 |

**标签与候选空间**：`1` 共 401 个，`-1` 共 11 个，`0` 共 92,151 个；其中 `0` 是未记录/未知，不能当作已证实无效。613 种药物中有 204 种至少涉及一条非零关联；151 种疾病中有 116 种至少涉及一条非零关联。报告必须同时展示总候选空间、有已知关联的实体数、实际进入评价的病例数，避免将稀疏标签造成的覆盖差异隐藏在一个分数中。

**读取和质控顺序**：

1. 读取三个 CSV，保留首列为原始字符串 ID，不自动转成数字；
2. 验证 `ratings_mat` 的药物集合等于 `items` 的药物列集合、疾病集合等于 `users` 的疾病列集合；
3. 验证 `items` 和 `users` 的基因集合及顺序；若顺序不同则以基因 symbol 显式重排，禁止按位置直接相乘；
4. 检查重复基因 ID、重复药物/疾病 ID、NaN、Inf、全零向量和常量向量；
5. 记录每条药物—疾病比较实际使用的非缺失交集基因数；
6. 把 613×151 分数矩阵按 `ratings_mat` 的行列顺序写出，另存映射清单和数据哈希；
7. 对全零或不可计算的组合写入预先规定的缺失策略，不因测试标签而补值或剔除。

**最小计算示例**：对于疾病 `C...` 和药物 `DB...`，取 `users.csv` 与 `items.csv` 中相同的基因行，计算 `-Spearman(disease_vector, drug_vector)`；越大表示表达变化越相反。该分数只表达“签名反转”，并不等价于药物在患者中有效。对所有组合批量计算后，按照官方接口提交矩阵。

**外部评测接法与仍需完成的验证**：RECeSS 提供 `random_simple` 与 `weakly_correlated` 拆分以及官方基线，项目应复用其种子、划分函数和评分器。我们自己的无训练适配器需要先跑通官方 Demo、核对 `fit/predict_proba` 输入输出、检查其对 `0` 与 `-1` 的实际评分处理，再宣称与官方实验完全兼容。当前已确认数据文件存在并已核验维度；适配器尚未实现和运行，计划中的成绩都是目标而非现有结果。

**稀疏标签的指标限制**：只有 11 个显式失败关联，许多疾病缺少足够正负例，按疾病计算 AUC 可能不可定义或不稳定。应先复现官方指标，再单独报告“可计算的疾病数/药物数”、覆盖率与补充排名指标；不得把 92,151 个未知关系统称负例并报告看似很高的 Accuracy。

## 5.7 GSE32863：LUAD 患者疾病签名

**原始来源**：[NCBI GEO GSE32863](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32863)。研究为人肺腺癌肿瘤与邻近非肿瘤肺组织的 Illumina HumanWG-6 v3.0 芯片表达分析。GEO 网页摘要写“60 对”，但 Overall Design 与公开 GSM 样本清单对应 **58 对表达样本（116 份）**；本项目以实际 GSM、患者 ID 和分组核验结果为准。一个 [GEO 样本页](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM813463) 标明 `VALUE` 为 robust spline normalized、log2 transformed。该队列是 LUAD 案例输入，并非 TRANSCRIPT Benchmark 的原始组成样本。

**2026-09-24 实际核验补充**：已下载 Series Matrix 的 116 份表达样本并解析标题。按标题中的患者编号只能形成 **57 个完整 Tumor/Normal 配对**；`GSM813507` (`3023_T`) 与 `GSM813518` (`3035_N`) 各自缺少相同编号的另一组。它们不能凭相邻位置强行配对；当前分析明确排除这 2 份样本，使用 57 对。详见 `docs/luad_data_audit.md` 和数据 manifest。

**下载内容**：Series Matrix、样本元数据、平台注释 GPL6884；必要时取 GEO 提供的非归一化补充文件复核。先使用已经说明归一化及 log2 的处理后表达值，不能再盲目做一遍 log2。下载后输出 `sample_manifest.tsv`，至少包含 `GSM`、原始样本标题、患者 ID、Tumor/Normal、是否成对、是否纳入及原因。

**处理步骤**：

1. 解析 GSM 标题中的 `_T` / `_N` 与患者编号，验证每位患者恰有一份肿瘤和一份邻近正常样本；
2. 用 GPL6884 将 Illumina 探针 ID 映射到基因 symbol；对一对多或多对一映射设定固定、可审计规则并记录丢失比例；
3. 检查分布、缺失值、离群样本和配对关系；若公开数据与 58 对不一致，停止并先解决差异；
4. 使用配对设计做差异表达，设计矩阵体现 `patient_id` 与 Tumor/Normal；芯片数据可使用 limma；
5. 使用 Benjamini–Hochberg 进行多重检验校正，输出全部基因的 log2FC、原始 p 值、FDR、样本数和探针处理规则；
6. 初始候选签名采用 FDR < 0.05 与 `|log2FC| ≥ 1`；若基因数量不足，改用预先定义的 top-N 敏感性分析，并同时保留原始阈值结果；
7. 形成疾病上调和下调基因集，固定文件、算法版本、参数与 SHA-256。

**产物**：`luad_samples.tsv`、`luad_deg_all.tsv`、`luad_up.txt`、`luad_down.txt`、`luad_qc.json`。报告要记录每一步剩余的样本、探针和基因数。

## 5.8 LINCS GSE92742：LUAD 药物扰动签名

**原始来源**：[Broad CLUE 官方 GEO 数据指南](https://clue.io/connectopedia/guide_to_geo_l1000_data)和 [NCBI GEO GSE92742](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742)。Broad 建议一般分析使用 Level 5 签名；其指南列出 Phase I Level 5 GCTX、`sig_info`、`gene_info`、`pert_info` 和校验文件。Level 5 是重复测定汇总后的药物扰动表达签名，矩阵包含 473,647 条签名与 12,328 个基因特征；完整文件约二十 GB 级，实际下载体积以 GEO 文件页为准。[官方术语表](https://clue.io/connectopedia/glossary)说明了 signature、perturbagen、时间、剂量等字段。

**最少需要的文件**：

- `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz`：表达签名；下载时以 GEO 页面实际文件名为准；
- `GSE92742_Broad_LINCS_sig_info.txt.gz`：`sig_id`、`cell_id`、`pert_id`、`pert_type`、时间/剂量等列；
- `GSE92742_Broad_LINCS_gene_info.txt.gz`：基因特征注释；
- `GSE92742_Broad_LINCS_pert_info.txt.gz`：化合物注释；
- `GSE92742_SHA512SUMS.txt.gz`：发布方校验和。

**可执行的子集提取**：

1. 先下载体积较小的元数据文件，读取实际表头和各条件计数；
2. `cell_id == A549`，并按 [Broad 定义](https://clue.io/connectopedia/perturbagen_types_and_controls)选择 `pert_type == trt_cp`；
3. 从所选行取得 `sig_id` 列表，再通过 GCTX 的按列/签名选择功能提取数据，避免将全部矩阵展开到内存；
4. 检查 `pert_itime`、`pert_idose`、签名质量字段的实际存在性及分布，先冻结主条件和备选条件，再执行排名；
5. 对同一 `pert_id` 的重复签名保留实验条件，按预先规定的稳健聚合方式生成药物级分数，同时记录异质性；
6. 与 GSE32863 疾病签名取基因交集，报告交集大小、是否只用 landmark/BING 以及各自结果；
7. 保存 A549 子集的 `sig_id`、药物 ID、时间、剂量、质量、基因空间和文件哈希。

**当前状态与算力门槛**：官方文件和元数据已确认存在，但本项目尚未下载完整 Level 5 文件，也尚未计算实际 A549 化合物数量。因此不预先声称“已有多少可用 A549 药物”。如果本地磁盘、网络或内存不足，优先做小型 TRANSCRIPT Benchmark；LUAD 案例可使用官方签名检索服务或明确记录的受限子集，但要标明与完整离线 LINCS 分析的差别。

## 5.9 Repurposing Hub、Open Targets、PubMed：证据怎样接入

| 来源 | 已确认入口 | 取得什么 | 系统内用途 | 注意 |
|---|---|---|---|---|
| [Broad Repurposing Hub](https://repo-hub.broadinstitute.org/repurposing) | 官方 Drug information 与 Sample information 批量下载；官方页面说明**没有**供批量注释使用的 API | 药物名称、临床阶段、作用机制、蛋白靶点；样品信息含 Broad ID、PubChem ID、SMILES 等 | LINCS 药物身份标准化、靶点和阶段注释 | 官方页面列出 2025-08-18 版本；保留实际下载版本。已知适应症字段只在 Research Open Mode 用，不能进入盲测特征 |
| [Open Targets Platform](https://platform-docs.opentargets.org/data-access/graphql-api) | GraphQL 查询单个疾病、靶点、药物和关联；大批量下载按 [官方 Parquet 说明](https://platform-docs.opentargets.org/data-access/datasets) | 疾病—靶点证据、药物靶点和来源记录 | 检查候选机制是否与疾病生物学相符，保留 evidence source 和版本 | Open Targets 关联分数不是临床成功概率，也不等于真实因果效应 |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov/download/) | NCBI E-utilities 搜索与抓取 PMID/摘要 | 原文出处、发表时间、题目、摘要和可用的研究类型 | 针对 Top-N 候选建立支持/反对证据清单，核对 PMID | 仅命中关键词不等于支持具体结论；关键论断需读摘要或可获取的正文 |

**统一 ID 链**：`LINCS pert_id / pert_iname → Repurposing Hub 标准名称或 PubChem ID → 药物名称、靶点 → Open Targets drug/target ID → PubMed PMID`。优先使用相同、稳定的数据库 ID 和 InChIKey，名称匹配只做候选；对盐型、同义词、同名异物和多成分药物要求人工复核。每一步保存 `source_id`、`target_id`、匹配规则、置信度和未匹配原因。没有可靠映射的药物仍保留原始计算分数，但不能自动补写靶点或临床信息。

**证据链输出**：每条重要结论保留原始源 URL、PMID/DOI、发表或数据版本日期、证据类型、具体支持的陈述以及相反证据。优先区分体外签名、动物实验、临床观察和正式试验，不把“文献提到该药”计为药效证据。

## 5.10 数据版本、派生文件与仓库规则

原始大数据与可能受附加条款约束的下载文件保存在本地 `data/raw/`，**GitHub 仓库只提交代码、配置、数据 Manifest、必要的小型示例和可重建的结果摘要**。每个数据集的 Manifest 至少包含：来源页面、直接下载地址、accession/DOI、版本、下载日期、原始文件名、字节数、SHA-256/发布方校验和、许可、预处理命令、纳入/排除数与生成物哈希。

建议最终派生文件：

```text
data/manifests/transcript-v2.0.0.yaml
data/manifests/gse32863.yaml
data/manifests/gse92742-a549.yaml
data/manifests/repurposing-hub.yaml
data/processed/transcript/score_matrix.parquet
data/processed/luad/deg_all.tsv
data/processed/luad/lincs_a549_signatures.parquet
artifacts/reports/luad_top10.json
artifacts/traces/<run_id>.jsonl
```

Manifest 和文件名是计划中的交付格式，当前尚未生成，不能写成已完成。

## 5.11 数据与任务的 Go/No-Go 检查表

| 检查 | 通过标准 | 未通过时的动作 |
|---|---|---|
| TRANSCRIPT 完整性 | MD5 正确、三矩阵实测为 613×151、12,096×613、12,096×151；标签 401/11/92,151 | 暂停 Benchmark，不使用不明版本 |
| RECeSS 适配 | 官方 Demo 跑通；自己的分数矩阵由官方评价代码成功读取；标签隔离成立 | 暂不宣称“已刷 RECeSS Benchmark” |
| LUAD 队列 | 58 对样本身份和分组逐一核对；每步样本数可追踪 | 停止差异表达，排查 GEO 元数据 |
| LINCS A549 | 存在可核验的 `trt_cp` 签名；时间、剂量、质量字段已核查；可提取固定子集 | 改用明确标注的小规模方案，不宣称完整 A549 覆盖 |
| 跨库映射 | 报告 LINCS → Hub → Open Targets 的成功率与歧义数 | 对未映射候选只输出原始分数，不自动补充机制 |
| 文献支持 | 每个 Top-10 候选有可复核 PMID/DOI 或明确写“暂无直接证据” | 降低证据等级，不能编造引用 |

---

## 6. 系统总体架构

```text
用户疾病任务
    ↓
LLM Planner：理解任务、制定计划、生成开放式说明
    ↓
Jev Choice/Noul：选择有限操作、检查是否满足执行条件
    ↓
Data Tools：下载、解析、质控、ID 映射
    ↓
Deterministic Analysis：差异表达、反转分数、通路分析
    ↓
Jev Score/Choice：质量分级、下一步路由、置信度门控
    ↓
Evidence Tools：Open Targets、PubMed、药物元数据
    ↓
LLM：综合机制、生成候选说明
    ↓
Jev Noul/Score：证据充分性、风险和升级判断
    ↓
Code Validator：Schema、ID、分数、引用、权限硬校验
    ↓
报告 + Evidence Ledger + Trace + Benchmark Output
```

### 6.1 职责分离原则

| 组件 | 负责 | 不负责 |
|---|---|---|
| LLM | 开放式规划、解释、报告、复杂歧义处理 | 大规模数值计算、最终权限控制 |
| Jev | 封闭选项、评分、概率判断、置信度门控 | 长文本、代码、总结、开放式科研论证 |
| 确定性工具 | 统计分析、数据处理、排名计算 | 自主改变业务规则 |
| 普通代码 | 状态机、权限、阈值、Schema、日志 | 模糊语义判断 |
| 外部 Evaluator | Benchmark 最终评分 | 指导系统迭代 |

---

## 7. Jev 模型的正确定位与接入计划

## 7.1 已确认信息

Jev 是 TypeSafe AI 于 2026-09-15 发布并提供早期访问的首个 System One 模型。它不是聊天模型，输入状态和预定义问题，输出可直接供程序使用的类型化决策、概率和置信度。

Jev 支持三类问题：

- **Choice**：从给定选项中选择；
- **Score**：按有序等级评分；
- **Noul**：返回某个陈述为真的概率。

官方 API 提供 `/v1/systemone`，可通过 `/v1/models` 查询可用模型名称。官方宣传的速度、价格和相对性能属于供应商数据，项目中必须通过自己的 Internal Eval Suite 独立测量，不能直接当作实验结论。

在本文档和代码中统一写作 **Jev**，避免与生物医学中的日本脑炎病毒缩写 JEV 混淆。

## 7.2 Jev 在本项目中的四类节点

### A. 工具路由 Choice

输入当前 Agent 状态，从固定选项中选择：

```text
CONTINUE_ANALYSIS
RETRY_CURRENT_TOOL
USE_CACHED_DATA
CHANGE_ALLOWED_TOOL
REQUEST_HUMAN_REVIEW
STOP_WITH_INSUFFICIENT_DATA
```

Jev 只能从允许列表中选择，不能生成新的工具名称。

### B. 数据质量 Score

对标准化的数据质量摘要评分：

```text
0 = unusable
1 = weak
2 = acceptable
3 = strong
```

最终是否继续仍由普通代码根据概率和阈值决定。

### C. 证据充分性 Noul

示例：

```text
“该候选是否至少具有两个相互独立且可核验的支持证据？”
“该文献是否真正支持当前机制陈述？”
“当前输出是否需要人工复核？”
```

### D. 候选分级 Choice/Score

用于将候选分为：

```text
SUPPORTED
PROMISING_BUT_INCOMPLETE
CONFLICTING
INSUFFICIENT_EVIDENCE
EXCLUDE
```

Jev 不修改转录组原始分数，只影响证据层的分级、是否升级到人工复核以及报告中的置信度表达。

## 7.3 Jev 置信度门控

示例策略：

```text
confidence >= T_high
    执行 Jev 选择的允许操作

T_low <= confidence < T_high
    交给通用 LLM 复核

confidence < T_low
    转人工或保守停止
```

`T_high` 和 `T_low` 只能根据内部开发 Eval 确定，不能根据外部 Benchmark 测试分数调节。

## 7.4 Jev 降级与替代方案

Jev 当前是闭源托管 API 和外部依赖，因此必须设计 feature flag：

```text
TYPESAFE_JEV_ENABLED=true|false
```

发生以下情况时自动降级：

- API 无法访问；
- 超时；
- 模型版本不可用；
- 置信度不足；
- 输入超出允许范围；
- 成本预算耗尽。

降级顺序：

```text
Jev → 固定规则 → 通用 LLM 结构化输出 → 人工复核
```

具体顺序根据节点风险确定。高风险节点不得在失败后默认放行。

## 7.5 Jev 专项内部指标

- Choice Accuracy；
- Noul AUROC；
- Brier Score；
- Expected Calibration Error；
- Score 与人工等级的 Spearman 相关；
- 决策覆盖率；
- 低置信度识别率；
- 错误自动执行率；
- 升级人工比例；
- P50/P95 延迟；
- 单任务成本；
- Schema/type 错误率；
- API 可用率；
- 与通用 LLM 判断的一致率和差异案例。

## 7.6 Jev 消融实验

| 实验 | 决策层 |
|---|---|
| J0 | 固定规则 |
| J1 | 通用 LLM 结构化输出 |
| J2 | Jev，无置信度升级 |
| J3 | Jev + 置信度门控 |
| J4 | Jev + 低置信度交给通用 LLM |

比较：

- 决策准确率；
- 校准程度；
- 工具选错率；
- 端到端成功率；
- 成本；
- 延迟；
- 人工升级率。

这部分是项目区别于普通生物信息学 Agent 的重要扩展，但不能替代主药物重定位实验。

---

## 8. 核心 Agent 模块

## 8.1 Planner

职责：

- 识别疾病、物种、组织和实验目标；
- 判断 Benchmark Strict Mode 或 Research Open Mode；
- 确定所需数据与工具；
- 产生初始执行计划；
- 设置允许和禁止的数据源；
- 估计数据规模、成本和风险。

Planner 输出必须满足固定 Schema，之后才交给 Jev 做有限路由判断。

## 8.2 Data Agent / Data Tools

职责：

- 下载或加载数据；
- 验证文件哈希；
- 解析样本元数据；
- 检查分组、缺失值和重复样本；
- 映射基因、药物和疾病 ID；
- 保存固定数据快照；
- 生成机器可读 QC 摘要。

## 8.3 Disease Signature Tool

肺腺癌案例首先采用 GSE32863：

- 实际表达分析样本按 GEO 实验设计为 58 对，即 58 个肿瘤和 58 个邻近正常样本；
- GEO 摘要提到 60 对，但实验设计及公开样本清单列出 58 对，下载后必须逐一核对样本 ID；
- 优先检查并使用配对设计；
- 如果配对关系无法可靠恢复，则采用非配对模型并明确记录。

输出字段：

- gene symbol；
- log fold change；
- p-value；
- adjusted p-value；
- regulation direction；
- quality flags；
- inclusion reason。

## 8.4 Perturbation Retrieval Tool

数据来源：

- TRANSCRIPT 药物表达矩阵；
- LINCS L1000；
- Repurposing Hub 元数据。

LUAD 案例优先提取 A549 子集：

1. 只保留化合物扰动；
2. 固定时间、剂量和质量筛选规则；
3. 保存筛选后的数据快照与哈希；
4. 不得在查看候选排名后调整筛选条件。

## 8.5 Deterministic Ranker

至少实现三类分数。

### Spearman 反转

```text
reversal_score = -Spearman(disease_vector, drug_vector)
```

### Rank-based Connectivity

- 疾病上调基因在药物下调端的富集；
- 疾病下调基因在药物上调端的富集。

### Pathway Antagonism

- 疾病激活通路与药物抑制通路的匹配；
- 疾病抑制通路与药物激活通路的匹配。

### 排名融合

使用 Reciprocal Rank Fusion：

```text
RRF(drug) = Σ 1 / (k + rank_method(drug))
```

初始固定 `k=60`。任何修改只能通过内部 Eval 决定。

## 8.6 Evidence Agent

对 Top-N 候选检索：

- 标准药物 ID；
- 靶点和作用机制；
- 相关疾病通路；
- 原适应症；
- 临床阶段；
- 已知安全性问题；
- 支持文献；
- 反对文献；
- 证据时间和证据类型。

## 8.7 Deterministic Validator

最终硬校验不能交给 Jev 或 LLM：

- Schema 校验；
- ID 是否存在；
- 分数和排名是否一致；
- PMID/DOI 格式与可解析性；
- 文件哈希；
- 工具权限；
- Benchmark 标签隔离；
- 运行预算；
- 最大重试次数。

---

## 9. Evidence Ledger

每个候选药物必须生成证据账本：

| 字段 | 内容 |
|---|---|
| drug_name | 标准药物名称 |
| drug_id | DrugBank、PubChem 或 ChEMBL ID |
| disease_id | 标准疾病本体 ID |
| transcriptomic_rank | 转录组排名 |
| reversal_scores | 各方法原始分数 |
| perturbation_context | 细胞系、剂量、时间 |
| targets | 标准靶点列表 |
| mechanism | 机制说明 |
| supporting_evidence | 支持证据与 PMID/DOI |
| contradicting_evidence | 冲突或失败证据 |
| evidence_types | 体外、动物、临床、数据库 |
| jev_decisions | Jev 问题、概率、置信度和版本 |
| confidence_tier | 候选分级 |
| limitations | 数据缺口与外推限制 |
| trace_id | 对应执行轨迹 |

所有 Jev 决策都必须保存：

- 模型实际名称；
- API 版本；
- 问题类型；
- 输入状态哈希；
- 可选答案；
- 返回概率；
- 置信度；
- 最终代码分支；
- 是否被 LLM 或人工覆盖。

---

## 10. 内部 Eval Suite

目录统一命名：

```text
evals/internal/
```

## 10.1 Tool Contract Evals

至少 20 个确定性用例：

- GEO 元数据解析；
- 样本分组；
- 基因 ID 映射；
- 差异表达输出；
- LINCS 条件过滤；
- Open Targets 响应解析；
- PubMed PMID 验证；
- 分数方向；
- RRF 排名；
- 缺失值和重复 ID。

## 10.2 Workflow Evals

至少 15 个流程用例：

- 正常任务；
- 疾病名称歧义；
- 找不到表达数据；
- 药物缺少扰动签名；
- API 超时；
- 数据字段变化；
- 文件损坏；
- 工具返回空结果；
- 执行中断后恢复；
- 成本预算耗尽。

## 10.3 Scientific Consistency Evals

至少 15 个用例：

- 疾病上调与药物下调方向；
- 配对设计选择；
- 多重检验校正；
- 通路方向解释；
- 排名与分数一致性；
- 未知关联不能解释为负例；
- 体外证据不能写成临床疗效。

## 10.4 Evidence Evals

至少 10 个经过人工复核的用例：

- PMID/DOI 是否存在；
- 引用是否支持当前陈述；
- 是否混淆药物、靶点或疾病；
- 是否区分证据等级；
- 是否报告冲突证据；
- 是否存在无支持的机制描述。

## 10.5 Jev Decision Evals

至少 30 个封闭决策用例：

- 下一工具选择；
- 数据是否足够；
- 是否应该重试；
- 是否升级人工；
- 证据质量等级；
- 候选置信度等级；
- 是否存在明显冲突；
- 是否允许发布。

每个用例必须具有固定选项、参考决策和风险等级。

## 10.6 内部总体指标

| 指标 | 用途 |
|---|---|
| Task Completion Rate | 端到端完成率 |
| Tool Success Rate | 工具成功率 |
| Schema Validity | 输出结构正确率 |
| Citation Validity | 引用真实性 |
| Citation Entailment | 引用支持程度 |
| Evidence Coverage | 关键结论证据覆盖 |
| Unsupported Claim Rate | 无依据陈述比例 |
| Candidate Traceability | 候选可追溯程度 |
| Ranking Reproducibility | 排名稳定性 |
| Recovery Rate | 故障恢复能力 |
| Decision Calibration | Jev 概率校准 |
| Escalation Precision | 升级是否合理 |
| Latency | 执行时间 |
| Cost | API 与计算成本 |

## 10.7 建议验收门槛

以下是目标，不是已取得结果：

- Schema 有效率：100%；
- 确定性工具测试通过率：不低于 99%；
- 端到端任务完成率：不低于 90%；
- 引用有效率：不低于 95%；
- 人工抽查证据支持率：不低于 90%；
- 关键无支持陈述：不高于 5%；
- 固定配置下排名 Spearman 相关：不低于 0.95；
- 高风险 Jev 低置信度决策自动执行率：0%；
- 所有失败具有错误码、轨迹和可定位原因。

---

## 11. 外部实验矩阵

| 版本 | 系统配置 | 目的 |
|---|---|---|
| B0 | 随机排序 | 最低基线 |
| B0p | 训练折药物流行度（degree） | 必须报告的“笨基线”，预实验 AUC≈0.73 |
| B1 | 单一 Spearman 反转 | 单工具基线 |
| B1k | 药物/疾病表达相似度 kNN 标签传播（无训练） | 预实验 AUC≈0.65 |
| B2 | 多分数 + RRF（反转 + kNN + 流行度） | 确定性融合基线 |
| R* | RECeSS 官方基线（至少 3 个快速算法，如 PMF/LogisticMF/ALS-WR/BNNR） | 由我们在相同拆分上重跑，作为“SOTA 参照线” |
| B3 | 普通 ReAct Agent + 相同工具 | 非结构化 Agent 基线 |
| B4 | 状态机 Agent，无 Jev | 主 Agent 基线 |
| B5 | 状态机 Agent + Jev | 检验 Jev 决策层 |
| B6 | Agent + Jev + 置信度门控 | 检验安全升级策略 |
| B7 | 经受约束 RSI 改进后的冻结版本 | 最终系统 |

### 11.1 主消融

1. 去掉通路分析；
2. 去掉文献证据；
3. 去掉确定性 Validator；
4. 只使用单一反转分数；
5. RRF 与固定加权比较；
6. ReAct 与状态机比较；
7. 通用 LLM 决策与 Jev 决策比较；
8. Jev 无门控与 Jev 置信度门控比较；
9. 开放网络与固定数据快照比较。

### 11.2 解释规则

- Benchmark 分数提升只能说明已知关联恢复能力提高；
- 不能证明新候选具有真实临床疗效；
- Jev 带来的收益需要同时报告准确率、成本、延迟和错误执行率；
- 如果 Jev 只降低成本但降低高风险决策质量，不能视为成功；
- 如果只在内部 Eval 提升、外部 Benchmark 不提升，应明确报告差异。

---

## 12. 数据泄漏控制

## 12.1 目录隔离

```text
benchmark/input/       Agent 可访问
benchmark/labels/      仅 Evaluator 可访问
benchmark/output/      Agent 写预测结果
benchmark/results/     Evaluator 写评分结果
```

## 12.2 进程隔离

- Agent 进程无权读取 `benchmark/labels/`；
- Evaluator 在 Agent 完成后单独启动；
- Benchmark Strict Mode 禁止开放式联网搜索；
- 所有工具调用写入不可变日志；
- Jev 的输入不得包含测试标签或 evaluator 输出。

## 12.3 版本隔离

- 内部 Eval 用于开发；
- 外部测试集用于冻结版本的最终评测；
- 查看外部测试结果后发生的修改必须进入新实验轮次；
- 报告所有尝试过的正式版本，避免只展示最优结果。

---

## 13. LUAD 端到端案例

## 13.1 输入

- 疾病：肺腺癌；
- 疾病表达数据：GSE32863；
- 药物扰动数据：LINCS A549 子集；
- 药物元数据：Repurposing Hub；
- 靶点和疾病机制：Open Targets；
- 文献证据：PubMed。

## 13.2 执行步骤

1. LLM Planner 解析疾病和研究目标；
2. Data Tool 加载 GSE32863；
3. 确定性程序检查 58 对肿瘤与邻近正常样本，并以实际 GSM 清单核实配对；
4. Jev 对 QC 摘要给出质量 Score，并返回置信度；
5. 代码根据门槛决定继续、复核或停止；
6. 差异表达工具构建疾病签名；
7. 加载固定的 A549 药物扰动子集；
8. 计算 Spearman、Connectivity 和通路反转分数；
9. 使用 RRF 生成初始 Top-20；
10. Evidence Agent 检索靶点、机制、支持和反对证据；
11. Jev 对证据充分性进行 Noul 判断，对证据等级进行 Score；
12. 低置信度和高风险候选交给通用 LLM 或人工复核；
13. Deterministic Validator 检查 ID、引用、Schema、排名和权限；
14. 输出 Top-10 和完整 Evidence Ledger；
15. 对已知治疗药物进行恢复分析；
16. 对未知候选只表述为待实验验证的研究假设。

## 13.3 最终候选报告字段

- 药物名称和 ID；
- 最终排名；
- 各反转分数；
- 细胞系、剂量和时间；
- 主要靶点；
- 候选机制；
- 支持证据；
- 反对证据；
- 安全性提示；
- Jev 证据评分与置信度；
- 最终置信度等级；
- 不确定性；
- 下一步实验建议。

---

## 14. 受约束 RSI 计划

RSI 仅作为后期扩展，不能在基础系统尚未稳定时加入。

```text
收集失败轨迹
    ↓
归类失败原因
    ↓
Agent 提出提示词、规则或路由修改
    ↓
在隔离分支中生成修改
    ↓
运行完整内部 Eval Suite
    ↓
检查质量、Jev 校准、成本和安全门槛
    ↓
全部通过 → 人工批准合并
任一关键门槛退化 → 自动回滚
```

允许自动修改：

- 提示词；
- 工具路由规则；
- Jev 问题的文字表达；
- 非安全关键阈值提案；
- 重试与缓存策略；
- 报告结构。

禁止自动修改：

- 外部 Benchmark 标签；
- Evaluator；
- 数据隔离规则；
- 安全关键门槛；
- 最大预算；
- 人工审批要求；
- 已冻结正式测试结果。

Jev 可用于 RSI 中的变更风险分级和发布门控，但不能自行批准其自身规则的修改。

---

## 15. 工程目录建议

```text
drug-repurposing-agent/
├── README.md
├── pyproject.toml
├── configs/
│   ├── benchmark_strict.yaml
│   ├── research_open.yaml
│   ├── luad.yaml
│   └── jev.yaml
├── src/
│   ├── agent/
│   │   ├── planner.py
│   │   ├── state.py
│   │   ├── workflow.py
│   │   └── policies.py
│   ├── jev/
│   │   ├── client.py
│   │   ├── questions.py
│   │   ├── gating.py
│   │   └── fallback.py
│   ├── tools/
│   │   ├── geo.py
│   │   ├── differential_expression.py
│   │   ├── lincs.py
│   │   ├── open_targets.py
│   │   ├── pubmed.py
│   │   └── id_mapping.py
│   ├── ranking/
│   │   ├── correlation.py
│   │   ├── connectivity.py
│   │   ├── pathway.py
│   │   └── rrf.py
│   ├── evidence/
│   │   ├── ledger.py
│   │   └── citation_validator.py
│   └── observability/
│       ├── trace.py
│       ├── cost.py
│       └── metrics.py
├── benchmarks/
│   ├── recess_adapter/
│   ├── repodb_adapter/
│   └── bioagent_adapter/
├── evals/
│   └── internal/
│       ├── tool_contracts/
│       ├── workflows/
│       ├── scientific/
│       ├── evidence/
│       ├── jev_decisions/
│       └── robustness/
├── data/
│   ├── manifests/
│   ├── raw/
│   ├── processed/
│   └── snapshots/
├── benchmark/
│   ├── input/
│   ├── labels/
│   ├── output/
│   └── results/
├── tests/
├── artifacts/
│   ├── traces/
│   ├── reports/
│   └── benchmark_results/
└── docs/
    ├── architecture.md
    ├── evaluation_protocol.md
    ├── data_card.md
    └── limitations.md
```

---

## 16. 八周实施时间表

## 第 1 周：Benchmark 和数据落地

任务：

- 下载 TRANSCRIPT v2.0.0；
- 保存 DOI、版本和 MD5；
- 克隆 RECeSS Benchmark 代码；
- 运行官方 Demo；
- 创建隔离环境；
- 建立数据 Manifest；
- 隔离 Benchmark 标签。

验收：

- 官方 Demo 可运行；
- 三个 TRANSCRIPT 矩阵维度正确；
- Agent 无法访问测试标签；
- 环境可复现。

## 第 2 周：确定性分析工具

任务：

- 实现 ID 和基因向量对齐；
- 实现 Spearman 反转；
- 实现 Rank-based Connectivity；
- 实现 RRF；
- 生成完整分数矩阵；
- 建立工具单元测试。

验收：

- 固定输入产生固定输出；
- 613×151 完整分数矩阵可生成，并另报 204×116 已知关联实体覆盖情况；
- 无维度错位或隐式丢失；
- 所有分数可追溯。

## 第 3 周：无 Jev 的 Agent MVP

任务：

- 定义 Agent State；
- 实现 Planner；
- 封装工具；
- 实现状态机；
- 加入缓存、重试和错误码；
- 输出结构化结果。

验收：

- 单命令完成 TRANSCRIPT 分析；
- 工具失败可以恢复；
- 输出满足 Schema；
- 保存完整 Trace。

## 第 4 周：LUAD 端到端案例

任务：

- 处理 GSE32863；
- 验证样本设计；
- 构建 LUAD 表达签名；
- 提取 LINCS A549 子集；
- 生成 Top-20 初始候选；
- 固定所有筛选参数。

验收：

- 输入、参数和结果可追溯；
- 候选可还原到具体分数；
- 无人工复制中间结果。

## 第 5 周：证据层和内部 Eval

任务：

- 接入 Open Targets；
- 接入 PubMed；
- 生成 Evidence Ledger；
- 实现引用验证；
- 建立至少 60 个非 Jev 内部 Eval。

验收：

- Top-10 均有结构化证据；
- 支持和反对证据分开；
- 内部 Eval 可自动运行；
- 关键失败可以定位。

## 第 6 周：Jev 接入与专项 Eval

任务：

- 接入官方 `/v1/systemone` API；
- 运行时查询实际模型名称；
- 实现 Choice、Score、Noul 封装；
- 实现置信度门控和降级；
- 建立至少 30 个 Jev 决策 Eval；
- 比较规则、通用 LLM 和 Jev。

验收：

- Jev 关闭时系统仍可运行；
- 所有决策有概率和 Trace；
- 高风险低置信度不会自动执行；
- 得到独立的准确率、校准、延迟和成本数据。

## 第 7 周：外部 Benchmark 和消融

任务：

- 接入 RECeSS Evaluator；
- 运行 B0 至 B6；
- 完成主要消融；
- 审计 repoDB 覆盖率；
- 选择性运行 BioAgent Bench。

验收：

- 输出正式 Benchmark 指标表；
- 所有版本资源预算一致；
- 保留原始分数和配置；
- 不只保留最好结果。

## 第 8 周：受约束改进与最终交付

任务：

- 进行一次受约束 RSI 实验；
- 运行全量内部 Eval；
- 冻结最终版本；
- 完成最后一次外部 Benchmark；
- 制作报告、PPT 和演示视频。

验收：

- 展示一次失败—修改—评测—接受/回滚流程；
- 最终成绩来自冻结版本；
- 局限性和失败案例完整披露；
- 项目可在新环境中复现。

---

## 17. Go/No-Go 门槛

### Gate 1：Benchmark 可复现

- 数据可下载；
- 哈希一致；
- 官方 Demo 成功；
- 标签隔离完成。

### Gate 2：科学计算正确

- 基因方向正确；
- ID 无错位；
- 分数符号正确；
- 排名可以人工抽查复算。

### Gate 3：端到端闭环

必须完成：

```text
疾病输入 → 数据 → 计算 → 排名 → 证据 → 验证 → 报告
```

### Gate 4：内部可靠性达标

- 核心 Eval 达到预设门槛；
- 失败有可解释轨迹；
- 无严重引用和权限问题。

### Gate 5：Jev 可安全降级

- API 不可用时主系统不中断；
- 低置信度自动升级；
- Jev 不能绕过确定性安全规则。

### Gate 6：冻结后外部评测

- 版本、提示词、Jev 问题和阈值全部冻结；
- 再运行正式 Benchmark；
- 测试结果不反馈到同一版本。

---

## 18. 主要风险与处理方案

| 风险 | 处理方案 |
|---|---|
| RECeSS 主要评价最终排序而非完整 Agent 过程 | 用 RECeSS 评价输出，用 BioAgent Bench 和内部 Eval 评价执行过程 |
| Benchmark 标签泄漏 | 目录、进程、工具和网络四层隔离 |
| 未知关联不是负例 | 优先使用排名指标，限制结论范围 |
| LINCS 数据过大 | 固定提取 A549 子集或优先使用 TRANSCRIPT |
| 细胞系不能代表患者 | 明确证据层级，不宣称临床有效 |
| 文献引用幻觉 | PMID/DOI 硬验证与人工抽查 |
| Agent 输出不稳定 | 状态机、确定性工具、低温度与重复运行 |
| Jev 是新模型且处于早期访问 | Feature flag、降级策略、版本记录和独立 Eval |
| Jev 为闭源托管服务 | 不把核心统计能力绑定在 Jev 上 |
| Jev 供应商性能宣传尚缺少充分独立复现 | 自测准确率、校准、延迟、成本和失败率 |
| Jev 判断错误但置信度高 | 高风险节点增加硬规则、LLM/人工复核和错误执行指标 |
| Jev 与日本脑炎病毒 JEV 混淆 | 文档和代码统一写作 Jev，并注明 TypeSafe AI |
| RSI 过拟合内部 Eval | 保留冻结验收集，外部测试仅在版本冻结后运行 |

---

## 19. 最终交付物

1. 可运行的药物重定位 Agent；
2. RECeSS/TRANSCRIPT Benchmark 适配器；
3. 外部 Benchmark 成绩表；
4. 内部 Eval Suite；
5. Jev Choice、Score、Noul 接入层；
6. Jev 置信度门控和降级机制；
7. Jev 与规则、通用 LLM 的对比实验；
8. LUAD 端到端案例；
9. Top-10 候选药物证据报告；
10. Evidence Ledger；
11. 完整 Agent Trace 和成本报告；
12. 消融实验；
13. 一次受约束 RSI 演示；
14. Docker 或锁定环境；
15. 数据卡、系统卡和限制说明；
16. 课程设计报告；
17. 答辩 PPT；
18. 演示视频。

---

## 20. 答辩叙事

建议用以下主线讲述：

> 我们开发了一个不依赖新预测模型训练的药物重定位 Agent。系统由通用 LLM、Jev System One 决策模型、确定性生物信息学工具和普通控制代码组成：通用 LLM 负责开放式规划和解释，Jev 负责高频且答案空间封闭的结构化决策，确定性工具负责科学计算，代码负责权限、阈值和验证。系统在 RECeSS 公开药物重定位 Benchmark 的 TRANSCRIPT 数据集上接受外部评测，并使用内部 Eval Suite 完成开发、回归、Jev 校准和受约束改进。最后，以肺腺癌为案例展示从疾病表达数据到候选药物证据报告的完整流程。

需要强调：

- 外部 Benchmark 是别人的；
- Internal Eval Suite 是我们自己的，但不称为 Benchmark；
- Jev 是一个新发布的结构化决策模型，不是自定义验证模块缩写；
- Benchmark 分数反映已知关系恢复能力，不代表临床疗效；
- 项目价值既包括生物医学结果，也包括可迁移到企业 Agent 的工程方法。

---

## 21. 已确认资源与参考链接

### 外部 Benchmark 与数据

- RECeSS Benchmark 代码：<https://github.com/RECeSS-EU-Project/benchmark-code>
- RECeSS Benchmark 论文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC11751339/>
- TRANSCRIPT v2.0.0：<https://zenodo.org/records/7982976>
- repoDB Benchmark 研究：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7153111/>
- BioAgent Bench：<https://github.com/bioagent-bench/bioagent-bench>
- RepurposingBench：<https://atlasdiscovery.bio/repurposingbench>

### Jev 官方资源

- TypeSafe AI Jev 发布说明：<https://typesafe.ai/blog/introducing-system-one-models-and-jev>
- TypeSafe AI 首页：<https://typesafe.ai/>
- TypeSafe API 文档：<https://api.typesafe.ai/docs>
- TypeSafe Workflow Evals：<https://evals.typesafe.ai/>

### 生物信息学数据源

- TRANSCRIPT 原始压缩包：<https://zenodo.org/records/7982976>
- RECeSS 数据构建 Notebook：<https://github.com/RECeSS-EU-Project/drug-repurposing-datasets>
- GSE32863 肺腺癌配对表达数据：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32863>
- GSE92742 LINCS Phase I：<https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742>
- Broad CLUE 数据文件指南：<https://clue.io/connectopedia/guide_to_geo_l1000_data>
- Broad 药物重定位元数据：<https://repo-hub.broadinstitute.org/repurposing>
- Open Targets GraphQL：<https://platform-docs.opentargets.org/data-access/graphql-api>
- Open Targets 批量下载：<https://platform-docs.opentargets.org/data-access/datasets>
- PubMed E-utilities：<https://pubmed.ncbi.nlm.nih.gov/download/>

---

## 22. 第一阶段立即执行清单

- [ ] 下载并校验 TRANSCRIPT v2.0.0；
- [ ] 复现 RECeSS 官方 Demo；
- [ ] 建立 Benchmark 标签隔离目录；
- [ ] 固定项目 Python 与依赖版本；
- [ ] 实现无 Agent 的确定性反转基线；
- [ ] 生成第一版 613×151 分数矩阵；
- [ ] 建立数据 Manifest；
- [ ] 实现 Agent State 和工具 Schema；
- [ ] 先完成无 Jev 的 Agent MVP；
- [ ] 申请或确认 TypeSafe API 访问；
- [ ] 建立 Jev 关闭时的 fallback；
- [ ] 为 Jev 准备第一批 30 个内部决策 Eval；
- [ ] 完成 GSE32863 样本元数据审计；
- [ ] 固定 LINCS A549 子集筛选协议。

---

## 23. 计划评审补充（2026-09-24）

### 23.1 TRANSCRIPT 预实验：实测数据

脚本：`pilot_transcript/probe.py`、`probe2.py`。数据为已校验 MD5 的 TRANSCRIPT v2.0.0。**这是我们自己的简化评测（非 RECeSS 官方拆分和评分器），数值只用于判断方向，不能写进正式成绩表。**

| 方法 | 设置 | AUC |
|---|---|---:|
| 随机 | 全矩阵，401 正 vs 其余 | 0.482 |
| −Spearman 反转（计划 B1） | 全矩阵，不用标签 | **0.483** |
| +Spearman（反号） | 同上 | 0.517 |
| \|Spearman\| | 同上 | 0.502 |
| Top-100 上/下调基因 Connectivity | 同上 | 0.497 |
| −Spearman，仅正例 vs 11 个显式负例 | 412 个已知关联 | 0.487 |
| 按疾病 AUC（114 个有正例的疾病） | 均值 / 中位数 | 0.473 / 0.483 |
| Recall@10（反转） | 按疾病平均 | 0.0075（随机期望 0.0163） |
| 训练折药物流行度 | 5 折 CV，测试正例 vs 未知 | **0.729 ± 0.035** |
| 药物相似度 kNN 标签传播（k=10） | 同上 | 0.590 ± 0.010 |
| 疾病相似度 kNN | 同上 | 0.569 ± 0.034 |
| 药物 kNN + 疾病 kNN（秩和） | 同上 | 0.648 ± 0.036 |
| 上一行 + 反转 | 同上 | 0.601 ± 0.026（加入反转后反而下降） |

补充事实：1 个疾病向量方差为 0（需走缺失策略）；每个疾病的正例数中位数为 1，四分位点为 [0, 1, 1, 3, 29]，按疾病计算的指标方差会很大。RECeSS 论文称 TRANSCRIPT 是 8 个数据集中“最难的”，Top-3 算法 NS-AUC 中位数约 0.68，并提示矩阵类方法可能受评测偏差影响。

**结论**：

1. 在 TRANSCRIPT 上，疾病—药物签名反转**没有可检测的信号**：两个符号方向、绝对值和 Connectivity 都在 0.50 附近。计划中“多种反转方法 + RRF 优于单一方法”（RQ3）在该数据集上几乎必然得到空结果。
2. **按原方案不可能刷出 SOTA**。连“训练折流行度”这种笨基线都比反转法高约 0.25 AUC。
3. TRANSCRIPT 上的有效信号来自**已知关联结构**（流行度 / 协同过滤），而不是反转假设。无训练的 kNN 标签传播能拿到 0.65，但仍低于流行度基线。

### 23.2 能否给老师展示 Benchmark 结果

**能，但要改叙事。** 课程要求 Problem → Data → Model → Benchmark → Biological interpretation，并明确“不只看 accuracy”。可以展示的真实结果：

1. **正式成绩表**（第 7 周，官方拆分与评分器）：B0 随机、B0p 流行度、B1 反转、B1k kNN、B2 融合，以及我们重跑的 ≥3 个官方基线 R*。每行都写均值 ± 标准差，并标出“是否使用训练标签”。
2. **核心发现（诚实的负结果）**：“CMap 式签名反转在 TRANSCRIPT 上无法恢复已知适应症；能恢复的信号主要来自关联结构。”这属于有价值的科学结论，本身就对应课程中的“实验严谨性”。
3. **Agent 的价值不在 Benchmark 分数上**，应在内部 Eval（可追溯、失败恢复、引用有效率）和 LUAD 案例（生物学解释）中体现。

**“刷 SOTA”的可行口径**：RECeSS 没有公开排行榜，论文也没有给出逐算法的 TRANSCRIPT 数值表，“SOTA”需要我们自己在同一拆分上重跑官方基线后才能定义。现实目标是：“无训练方法达到或接近官方最佳基线”，而不是“超越所有方法”。如果 B2 在官方评分器下能 ≥ R* 中的最佳者，才可以这样表述：*在 TRANSCRIPT 上，一个不训练模型的方法达到了与训练型 CF 方法相当的水平*。

### 23.3 计划中的结构性问题

| # | 问题 | 影响 | 修改 |
|---|---|---|---|
| 1 | 核心方法在主 Benchmark 上等于随机（23.1） | RQ3 与 B1/B2 的主叙事不成立 | 主表加入 B0p/B1k/R*；把 RQ3 改为“反转信号在 TRANSCRIPT 与 LUAD 案例中是否存在” |
| 2 | Strict Mode 下 Agent 只是调度一次批量矩阵计算，B3–B7 在 RECeSS 上的分数**必然与 B2 相同**（除非运行失败） | 第 11 节“外部实验矩阵”中 B3–B7 无法在外部 Benchmark 上区分 | 外部表只放 B0–B2 + R*；B3–B7 的比较移到内部 Eval（完成率、恢复率、成本），不要在外部表中列出一排相同数字 |
| 3 | 官方代码要求 Python 3.8，依赖 stanscofi/benchscofi；本机为 Python 3.13 | 第 1 周 Demo 可能卡在环境 | 用 conda 单独建 `py38` 环境；先只跑 1 个快速算法 × TRANSCRIPT × random_simple 验证链路 |
| 4 | 官方对 `0`（未知）在测试折中的处理尚未核实；只有 11 个 `-1` | AUC 定义不清，Accuracy/F-score 可能无意义 | 第 1 周读 stanscofi 的 fold/metric 源码并写入 `evaluation_protocol.md`；Accuracy 只作附录；F-score 的阈值必须在训练折上预先固定 |
| 5 | 11 个官方基线中部分为 GNN（HAN、NIMCGCN），算力和时间成本高 | 第 7 周跑不完 | 先选 3–5 个快速算法作 R*，其余引用论文图 3 的趋势，不报具体数值 |
| 6 | 范围过大：Jev、RSI、repoDB、BioAgent Bench、RepurposingBench 放在 8 周课设里 | 10 分钟汇报讲不完，核心链路可能烂尾 | 必做：TRANSCRIPT 正式表 + LUAD 案例 + 内部 Eval + Agent 主流程。可选（按顺序）：Jev 消融 → repoDB 覆盖子集 → RSI。BioAgent Bench 与 RepurposingBench 删除或放入“未来工作” |
| 7 | Jev 为 2026-09-15 发布的早期访问闭源 API，访问权限尚未确认 | 第 6 周整体受阻 | 第 1 周就申请；若 2 周内未获批，第 6 周改为“规则 vs 通用 LLM 结构化输出”（J0/J1），Jev 仅保留接口 |
| 8 | Jev Eval 只有 30 个用例，却要报告 AUROC、ECE、Brier | 统计量置信区间极宽，结论不可靠 | 扩到 ≥100 个（可由真实 Trace 半自动生成再人工标注），或只报告准确率 + bootstrap CI，ECE 作为附录 |
| 9 | LUAD 案例缺少预先定义的“已知有效药”阳性对照清单 | 第 13.2 步“恢复分析”无法客观评分 | 查看排名**前**冻结一份 LUAD 获批/指南药物清单（如 EGFR/ALK/KRAS/MET 抑制剂、铂类、培美曲塞、紫杉醇、多西他赛等），与 LINCS A549 化合物取交集后计算富集或 AUC，再用置换检验给出 p 值 |
| 10 | A549 为 KRAS 突变、EGFR 野生型细胞系 | EGFR-TKI 在 A549 上可能不反转，易被误读为方法失败 | 在局限性中写明；阳性对照按细胞系背景分层解释 |
| 11 | LINCS Level 5 约 20 GB；计划中的 978 landmark 与 GSE32863 全基因组交集规模未定 | 下载/内存风险；基因空间不一致 | 优先使用 CLUE/SigCom LINCS 在线签名检索，或只提取 A549 的列；主分析固定为 landmark 基因，BING 作敏感性分析 |
| 12 | 一次性验证：反转方向本身没有正对照 | 符号写反时难以发现 | 加入 Gate 2 用例：药物自身重复签名的相似度应 > 0；已知 CMap 经典对（如 HDAC 抑制剂签名之间）应高度一致 |
| 13 | 缺少汇报排练和图表清单 | 10 分钟汇报没有抓手 | 见 23.5 |
| 14 | 通用 LLM 型号、温度、预算和 API 费用未指定 | 可复现性和成本指标无从谈起 | 在 `configs/` 固定模型 ID、温度=0、每任务 token 上限；Trace 记录实际用量 |
| 15 | 团队分工与每周负责人缺失 | 并行度低，出现瓶颈 | 建议按 Benchmark/数据、LUAD 生物学、Agent 工程、汇报四条线分工 |

### 23.4 修订后的研究问题

- **RQ1**（不变）端到端能力；
- **RQ2'**：签名反转在 TRANSCRIPT（第三方标签）和 LUAD（预先冻结的阳性对照）上是否能恢复已知治疗关系？与利用关联结构的无训练方法相比如何？
- **RQ3'**：多信号融合（反转 + kNN + 流行度 + 通路）在官方评分下能否接近训练型 CF 基线？
- **RQ4**（降为可选）Jev 决策层的价值，仅在内部 Eval 上评价；
- **RQ5**（降为可选）受约束改进。

### 23.5 10 分钟汇报建议结构与图表

| 时长 | 环节 | 展示内容 |
|---|---|---|
| 1 min | Problem | 药物重定位的意义；课题三“Agent 自动完成分析”的要求 |
| 1.5 min | Data | TRANSCRIPT 613×151 与标签稀疏度热图；GSE32863 58 对；LINCS A549 子集规模 |
| 2 min | Model / System | 架构图（LLM / 工具 / Validator 职责分离）；一次真实运行的 Trace 截图 |
| 2.5 min | Benchmark | 正式成绩表（B0/B0p/B1/B1k/B2/R*，均值±SD）；“反转≈随机”的核心发现；内部 Eval 通过率 |
| 2 min | Biological interpretation | LUAD Top-10 表：分数、靶点、通路、PMID、证据等级；阳性对照富集结果 |
| 1 min | 局限与结论 | 细胞系≠患者、未知≠阴性、Benchmark≠临床疗效 |

### 23.6 修订后的第 1–2 周优先项（替换第 22 节前 6 项的顺序）

- [x] 下载并校验 TRANSCRIPT v2.0.0（已完成）；
- [x] 反转基线预实验（已完成，结果见 23.1）；
- [ ] 建立 Python 3.8 conda 环境，跑通 RECeSS 1 个快速算法 × TRANSCRIPT；
- [ ] 阅读 stanscofi 的 fold 与 metric 源码，写出 `0`/`-1` 处理规则；
- [ ] 按官方接口实现 B0/B0p/B1/B1k/B2 并提交官方评分器；
- [ ] 重跑 3–5 个官方基线作为 R*；
- [ ] 冻结 LUAD 阳性对照药物清单（在计算任何 LUAD 排名之前）；
- [ ] 提交 TypeSafe Jev 访问申请，设置 2 周截止日期。
