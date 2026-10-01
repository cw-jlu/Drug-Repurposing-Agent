# Top-10 候选的通路层面反转（v1）

2026-10-01。规则在计算任何药物分数之前提交（commit `37cac48`）。输入核对在评分前修订并单独提交（`7043665`）：重新生成的排名复现了冻结的 Top-10 与逐字节一致的阳性对照名次，但完整排名文件的哈希与冻结版本不一致（不同 NumPy 版本的浮点格式差异）。脚本：`python -m scripts.luad_pathway_reversal`；结果：`benchmark/results/luad_pathway_reversal_v1.json`；图：`docs/figures/fig8_top10_pathway_reversal.png`。

**方法**：对疾病签名中 FDR < 0.05 的 Hallmark 通路，取“通路成员 ∩ 该方向疾病差异基因 ∩ 匹配的 LINCS landmark 基因”（至少 5 个）。反转分数 = −方向 × 药物在这些基因上的平均签名值，正值表示药物把这些基因推回正常方向。每个 Top-10 药物在全部 4,920 个 A549 药物中的百分位为最终指标。L1000 只测 978 个 landmark 基因，因此 4 条上调、11 条下调的显著通路因基因不足被跳过（见 JSON 中 `skipped_pathways`）。

| 通路（方向） | landmark 基因数 | Top-10 平均百分位 |
|---|---:|---:|
| Glycolysis (肿瘤上调) | 8 | 0.98 |
| Hypoxia (肿瘤下调) | 5 | 0.95 |
| E2F Targets (肿瘤上调) | 9 | 0.94 |
| G2-M Checkpoint (肿瘤上调) | 11 | 0.93 |
| Estrogen Response Late (肿瘤上调) | 7 | 0.92 |
| TNF-alpha Signaling via NF-kB (肿瘤下调) | 11 | 0.87 |
| Estrogen Response Late (肿瘤下调) | 6 | 0.81 |
| Epithelial Mesenchymal Transition (肿瘤下调) | 6 | 0.69 |

**解读**

- Top-10 的反转主要集中在肿瘤上调的增殖（G2-M、E2F）与糖酵解基因，以及肿瘤下调的缺氧、TNF-α/NF-κB 基因；这说明排名靠前的药物主要是在把增殖与代谢程序往正常方向推。
- 少数例外：beclomethasone-dipropionate 在 Epithelial Mesenchymal Transition (肿瘤下调) 上仅 0.43；clocortolone-pivalate 在 Epithelial Mesenchymal Transition (肿瘤下调) 上仅 0.39；diflorasone 在 Estrogen Response Late (肿瘤下调) 上仅 0.41；diflorasone 在 Epithelial Mesenchymal Transition (肿瘤下调) 上仅 0.07。说明同一类别（糖皮质激素）内部在 EMT 等通路上的作用并不一致。
- TNF-α/NF-κB 在肿瘤中下调，“反转”意味着药物上调这些基因；这与糖皮质激素经典的抗炎/抑制 NF-κB 作用方向相反，需要谨慎解读：A549 中的 24 小时转录反应不等于体内免疫效应，且这里只用到 11 个 landmark 基因。
- **重要限制：这不是独立验证。** Top-10 本身就是按反转同一疾病签名选出来的，通路基因又取自同一签名，因此高百分位在很大程度上是选择的必然结果。本分析的价值在于说明反转信号由哪些生物学程序驱动，而不是证明药效。
