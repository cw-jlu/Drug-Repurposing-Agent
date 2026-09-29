# LUAD Top-10 多智能体证据审阅（v1）

后续对本版 65 条引文所做的[来源与药物指向预筛](evidence_scope_audit.md)将 23 条列为需要进一步核对；该结果不改写这里的冻结分级。

> **这是文献分诊（literature triage），不是疗效证据。** 分级只描述在 PubMed 中检索到、并且通过引用校验的文献现状，用来安排后续研究的优先级；不构成治疗建议，也不证明任何候选药物对肺腺癌有效。

运行日期：2026-09-28。模型：`deepseek-flash`（DeepSeek beta 端点，strict function calling）。
代码：`src/drug_repurposing_agent/multi_agent_review.py`；运行器：`scripts/run_multi_agent_review.py`。
结果：`benchmark/results/multi_agent_review_v1.json`。
入口 trace：`artifacts/multi_agent_review/traces/`；每次模型请求的 trace：`artifacts/multi_agent_review/provider_traces/`（均被 git 忽略）。
输入：`configs/luad_top10_evidence_v1.json`，其 SHA-256 记录在结果中。

## 流程

1. **Literature Agent**：沿用现有的 `PubMedClient`（NCBI E-utilities，限速 0.36 s），并扩展了 ESearch ID 检索和 EFetch 摘要获取。检索式为 `"<药名>"[Title/Abstract] AND (lung adenocarcinoma OR non-small cell lung cancer OR NSCLC OR A549)`，按相关度取前 8 篇，只保留有摘要的记录。这个 Agent 提出支持性论断，每条论断都必须对应一个 PMID，并附上摘要中的原句。
2. **Critic Agent**：另做一次检索，在同样的药名和肺癌词之外，再加上 resistance/toxicity/adverse/"no effect"/"did not"/failed/worse/reduced/survival 等负面词。对于糖皮质激素类候选药（通路中含 glucocorticoid），还会补充一次共享的类别级检索（糖皮质激素/地塞米松 + NSCLC + 耐药/免疫治疗，取前 5 篇）。Critic 要对每一条已验证的支持论断给出 stands/weakened/refuted 的判断，并列出反证，反证同样要附原文引用。
3. **确定性校验器**：引用的 PMID 必须出现在该 Agent 实际检索到的记录集合中；引用句在空白归一化后，必须是该 PMID 摘要的逐字子串，且长度至少 20 个字符。不合格的论断会被丢弃并计数。Critic 引用的、不存在的 claim_id 也会被丢弃。
4. **Coordinator Agent**：从 {SUPPORTED, PROMISING_BUT_INCOMPLETE, CONFLICTING, INSUFFICIENT_EVIDENCE} 中选出一个分级，并写一段简短理由（strict JSON）。之后执行确定性护栏，护栏**只能降级**：没有任何已验证的支持论断时一律为 INSUFFICIENT_EVIDENCE；SUPPORTED 至少需要 2 个未被驳斥、候选药物本身范围内的 PMID。如果某个候选药物既没有检索到记录，也没有任何论断，就直接判为 INSUFFICIENT_EVIDENCE，不调用模型。

## 结果

| 排名 | 候选 | 支持检索命中 / 有摘要 | 已验证支持论断 | Critic 判定 | 已验证反证 | 多智能体分级 | 现有单轮结论 |
| ---: | --- | --- | ---: | --- | ---: | --- | --- |
| 1 | hydrocortisone | 61 / 8 | 6 | 6 条均为 weakened | 6 | INSUFFICIENT_EVIDENCE | insufficient_evidence |
| 2 | beclomethasone-dipropionate | 7 / 7 | 6 | 5 条 weakened，1 条 stands | 7 | INSUFFICIENT_EVIDENCE | insufficient_evidence |
| 3 | clocortolone-pivalate | 0 / 0 | 0 | — | 4（类别级） | INSUFFICIENT_EVIDENCE | insufficient_evidence |
| 4 | mometasone | 6 / 6 | 6 | 6 条均为 weakened | 4 | INSUFFICIENT_EVIDENCE | insufficient_evidence |
| 5 | flumetasone | 0 / 0 | 0 | — | 3（类别级） | INSUFFICIENT_EVIDENCE | insufficient_evidence |
| 6 | BRD-K84203638 | 0 / 0 | 0 | — | 0 | INSUFFICIENT_EVIDENCE（确定性判定，未调用模型） | insufficient_evidence |
| 7 | hydrocortisone-hemisuccinate | 0 / 0 | 0 | — | 4（类别级） | INSUFFICIENT_EVIDENCE | insufficient_evidence |
| 8 | fluorometholone | 0 / 0 | 0 | — | 4（类别级） | INSUFFICIENT_EVIDENCE | insufficient_evidence |
| 9 | fluticasone | 24 / 8 | 6 | 4 条 weakened，2 条 stands | 7 | INSUFFICIENT_EVIDENCE | insufficient_evidence |
| 10 | diflorasone | 0 / 0 | 0 | — | 2（类别级） | INSUFFICIENT_EVIDENCE | insufficient_evidence |

**与现有单轮结论的比较**：10/10 一致，全部为 INSUFFICIENT_EVIDENCE。现有单轮结论来自 `configs/luad_top10_evidence_v1.json` 和 `docs/luad_top10_evidence_matrix.md`。没有任何一个分级是由护栏改出来的：每个调用了模型的候选，Coordinator 给出的都已经是 INSUFFICIENT_EVIDENCE。

各候选的主要理由（由 Coordinator 生成，已经过校验）：
- **hydrocortisone**：检索到的支持记录几乎都是免疫检查点抑制剂相关的垂体炎/肾上腺功能不全个案报告，氢化可的松在其中是替代治疗，不涉及抗肿瘤作用。
- **beclomethasone-dipropionate、mometasone、fluticasone**：支持论断全部来自 A549 细胞上的体外抗炎、转录抑制或 CYP3A5 诱导等替代指标，没有任何 LUAD/NSCLC 抗肿瘤终点、动物实验或临床证据。
- **其余 6 个**：按确切药名没有检索到候选药本身的文献，只有糖皮质激素类别级的负面信号（免疫治疗干扰、耐药）。

**幻觉引用拒绝率**：两个 Agent 共提出 65 条带引用的论断（支持 24 条，反证 41 条），校验器拒绝 **0 条（0.0%）**。在 strict 工具调用、并且把检索到的摘要全文提供给模型的条件下，DeepSeek 没有编造 PMID，也没有改写引文。

调用量：共 22 次模型调用（4 个有支持文献的候选各 3 次 Literature+Critic+Coordinator，5 个只有类别级记录的候选各 2 次 Critic+Coordinator，BRD-K84203638 为 0 次），全部成功，0 次重试。共约 6.3 万 token，估算费用为 0.014 美元（off-peak）至 0.028 美元（peak）。PubMed 请求另计，不收费。

## 局限与观察

- **逐字引用不等于引用正确**。校验器只能保证引文确实存在于该 PMID 的摘要中，无法保证引文能支持对应的论断。人工抽查发现：Critic 曾把一篇关于柚皮素纳米载体的 A549 摘要（PMID 33249629）标为 fluticasone 的 `candidate` 级反证；多次把一条勘误声明（PMID 41376996，Published Erratum，摘要只有一句“This corrects the article”）当作证据引用；还常用"X 是安全/有益的"这种稻草人式表述来写反证。所以反证数量不能当作真实的矛盾强度，下一步应该加入论断–引文蕴含（NLI）检查或人工复核。
- **检索式与既有矩阵不同**。hydrocortisone 在既有矩阵中的候选药本身文献（PMID 1533045、35676421）没有进入按相关度排序的前 8 篇；fluticasone 的预防性荟萃分析（PMID 35843928）也没有被检索到。beclomethasone 的两篇既有文献（12538830、24555085）都检索到了。这是一次有边界的检索快照，不能证明"不存在证据"。
- 只按确切药名检索，没有扩展同义词、盐或酯形式；身份问题（立体化学、剂型歧义）只作为 Coordinator 的输入，没有经过化学层面的核验。
- 只用了一个模型，每个候选只跑一轮，结果没有做重复性检验。

## 复现

```
python scripts/run_multi_agent_review.py   # 需要 .env 中的 DEEPSEEK_API_KEY；已完成的候选会从 artifacts/multi_agent_review/checkpoint_v1.jsonl 续跑
```
