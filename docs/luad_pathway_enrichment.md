# LUAD 疾病签名通路富集（v1）

2026-10-01。对 GSE32863 57 对配对差异表达的预设签名（FDR < 0.05 且 |log2FC| ≥ 1：上调 512、下调 749，共检测 19404 个基因）分别做过度表达分析。基因集为 Enrichr 提供的 MSigDB Hallmark 2020 与 KEGG 2021 Human，哈希记录在 `data/manifests/enrichr-gene-sets.json`。方法为单侧超几何检验，背景为同时出现在检测集与该库中的基因；每个库 × 方向内做 BH 校正，背景基因不足 10 个的集合不检验。脚本：`python -m scripts.luad_pathway_enrichment`；结果：`benchmark/results/luad_pathway_enrichment_v1.json`；图：`docs/figures/fig7_luad_pathways.png`。

显著通路数（FDR < 0.05）：Hallmark 上调 8、下调 15；KEGG 上调 2、下调 23。

## Hallmark：肿瘤中上调

| 通路 | 重叠/集合 | 富集倍数 | FDR | 代表基因 |
|---|---:|---:|---:|---|
| Estrogen Response Late | 25/196 | 2.92 | 4.0e-05 | AGR2, CDC20, CDH1, CXCL14, DNAJC12, GALE |
| G2-M Checkpoint | 24/192 | 2.87 | 4.4e-05 | AURKA, AURKB, BIRC5, CCNB2, CCNF, CDC20 |
| Glycolysis | 24/195 | 2.82 | 4.4e-05 | ABCB6, AGRN, AK4, ANGPTL4, AURKA, CHPF |
| E2F Targets | 22/194 | 2.60 | 3.4e-04 | AURKA, AURKB, BIRC5, CCNB2, CCNE1, CDC20 |
| KRAS Signaling Up | 21/196 | 2.46 | 9.6e-04 | ADAM8, ANGPTL4, BIRC3, CFB, CSF2RA, ERO1A |
| mTORC1 Signaling | 19/194 | 2.25 | 5.6e-03 | AK4, ATP2A2, AURKA, CCNF, DAPP1, ERO1A |
| Estrogen Response Early | 17/190 | 2.05 | 2.4e-02 | CANT1, ELF3, FHL2, KCNK5, KRT15, KRT19 |
| Epithelial Mesenchymal Transition | 17/198 | 1.97 | 3.3e-02 | COL11A1, COL1A1, COL1A2, COL3A1, COL5A2, COMP |

## Hallmark：肿瘤中下调

| 通路 | 重叠/集合 | 富集倍数 | FDR | 代表基因 |
|---|---:|---:|---:|---|
| TNF-alpha Signaling via NF-kB | 45/198 | 3.07 | 1.1e-10 | ACKR3, ATF3, BTG1, BTG2, CCL2, CCL5 |
| KRAS Signaling Up | 35/196 | 2.41 | 1.1e-05 | ALDH1A2, CA2, CCND2, CD37, CFH, CXCR4 |
| Epithelial Mesenchymal Transition | 35/198 | 2.39 | 1.1e-05 | ABI3BP, CD44, CD59, CXCL12, CXCL8, DAB2 |
| Inflammatory Response | 32/198 | 2.18 | 1.9e-04 | AQP9, BTG2, C5AR1, CALCRL, CCL2, CCL5 |
| Xenobiotic Metabolism | 29/197 | 1.99 | 2.2e-03 | ALDH2, AOX1, AQP9, ATOH8, CA2, CAT |
| Hypoxia | 28/192 | 1.97 | 2.7e-03 | ACKR3, ANXA2, ATF3, BTG1, CAV1, CXCR4 |
| Coagulation | 22/137 | 2.17 | 2.7e-03 | A2M, ANG, APOC1, C1QA, C1R, C2 |
| Apoptosis | 24/158 | 2.05 | 3.0e-03 | ATF3, BTG2, CASP1, CAV1, CCND2, CD14 |

## KEGG 补充

上调：Central carbon metabolism in cancer（FDR 4.1e-02）；ECM-receptor interaction（FDR 4.1e-02）。下调前五：Complement and coagulation cascades（FDR 3.0e-09）；Systemic lupus erythematosus（FDR 5.2e-07）；Pertussis（FDR 2.8e-06）；Osteoclast differentiation（FDR 4.7e-06）；Staphylococcus aureus infection（FDR 5.3e-06）。

## 解读与边界

- 上调侧集中在增殖（G2-M 检查点、E2F 靶基因）、代谢重编程（糖酵解、mTORC1、KEGG 癌症中心碳代谢）与细胞外基质（KEGG ECM-受体相互作用），与肺腺癌已知的肿瘤程序一致。
- 下调侧以 TNF-α/NF-κB、炎症反应、补体与凝血为主，KEGG 中多为免疫相关条目。这更可能反映邻近正常肺组织中免疫与间质细胞成分在肿瘤块中相对减少，而不是肿瘤细胞内这些通路被抑制；批量芯片数据无法区分细胞组成与细胞内调控。
- EMT 与 KRAS Signaling Up 同时出现在上调和下调两侧，说明这些集合包含方向不同的成员，不能简单解读为“整体激活”。
- 这是对疾病签名的描述性注释，不检验药物疗效；Top-10 候选在这些通路上的反转见 `docs/luad_pathway_reversal.md`（若已运行）。
