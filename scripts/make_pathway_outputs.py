"""Chinese doc for the LUAD pathway enrichment (reads the committed JSON)."""
import json
from pathlib import Path

import numpy as np

d = json.loads(Path("benchmark/results/luad_pathway_enrichment_v1.json").read_text(encoding="utf-8"))
H = d["libraries"]["MSigDB_Hallmark_2020"]
K = d["libraries"]["KEGG_2021_Human"]
sig = lambda rows, n: [r for r in rows if r.get("fdr", 1) < 0.05][:n]

# Figure 7 is drawn by scripts/make_figures.py --only fig7 (shared style).

def tab(rows):
    out = ["| 通路 | 重叠/集合 | 富集倍数 | FDR | 代表基因 |", "|---|---:|---:|---:|---|"]
    for r in rows:
        out.append(f"| {r['term']} | {r['overlap']}/{r['set_size']} | {r['fold_enrichment']:.2f} | {r['fdr']:.1e} | {', '.join(r['genes'][:6])} |")
    return "\n".join(out)

n = lambda lib, dr: len([r for r in lib[dr] if r.get("fdr", 1) < 0.05])
doc = f"""# LUAD 疾病签名通路富集（v1）

2026-10-01。对 GSE32863 57 对配对差异表达的预设签名（FDR < 0.05 且 |log2FC| ≥ 1：上调 {d['signature_sizes']['up']}、下调 {d['signature_sizes']['down']}，共检测 {d['signature_sizes']['tested']} 个基因）分别做过度表达分析。基因集为 Enrichr 提供的 MSigDB Hallmark 2020 与 KEGG 2021 Human，哈希记录在 `data/manifests/enrichr-gene-sets.json`。方法为单侧超几何检验，背景为同时出现在检测集与该库中的基因；每个库 × 方向内做 BH 校正，背景基因不足 10 个的集合不检验。脚本：`python -m scripts.luad_pathway_enrichment`；结果：`benchmark/results/luad_pathway_enrichment_v1.json`；图：`docs/figures/fig7_luad_pathways.png`。

显著通路数（FDR < 0.05）：Hallmark 上调 {n(H,'up')}、下调 {n(H,'down')}；KEGG 上调 {n(K,'up')}、下调 {n(K,'down')}。

## Hallmark：肿瘤中上调

{tab(sig(H['up'], 8))}

## Hallmark：肿瘤中下调

{tab(sig(H['down'], 8))}

## KEGG 补充

上调：{'；'.join(f"{r['term']}（FDR {r['fdr']:.1e}）" for r in sig(K['up'], 5))}。下调前五：{'；'.join(f"{r['term']}（FDR {r['fdr']:.1e}）" for r in sig(K['down'], 5))}。

## 解读与边界

- 上调侧集中在增殖（G2-M 检查点、E2F 靶基因）、代谢重编程（糖酵解、mTORC1、KEGG 癌症中心碳代谢）与细胞外基质（KEGG ECM-受体相互作用），与肺腺癌已知的肿瘤程序一致。
- 下调侧以 TNF-α/NF-κB、炎症反应、补体与凝血为主，KEGG 中多为免疫相关条目。这更可能反映邻近正常肺组织中免疫与间质细胞成分在肿瘤块中相对减少，而不是肿瘤细胞内这些通路被抑制；批量芯片数据无法区分细胞组成与细胞内调控。
- EMT 与 KRAS Signaling Up 同时出现在上调和下调两侧，说明这些集合包含方向不同的成员，不能简单解读为“整体激活”。
- 这是对疾病签名的描述性注释，不检验药物疗效；Top-10 候选在这些通路上的反转见 `docs/luad_pathway_reversal.md`（若已运行）。
"""
Path("docs/luad_pathway_enrichment.md").write_text(doc, encoding="utf-8")
Path("tests/test_luad_pathway_enrichment.py").write_text('''from scripts.luad_pathway_enrichment import ora


def test_ora_detects_planted_enrichment_and_skips_small_sets():
    background = {f"G{i}" for i in range(1000)}
    library = {"planted": [f"G{i}" for i in range(50)],
               "null": [f"G{i}" for i in range(500, 550)],
               "tiny": ["G1", "G2"]}
    selected = {f"G{i}" for i in range(40)} | {f"G{i}" for i in range(900, 960)}
    rows = {r["term"]: r for r in ora(selected, background, library)}
    assert "tiny" not in rows
    assert rows["planted"]["overlap"] == 40 and rows["planted"]["fdr"] < 1e-10
    assert rows["null"]["overlap"] == 0 and rows["null"]["p_value"] == 1.0
''', encoding="utf-8")
print("ok")
