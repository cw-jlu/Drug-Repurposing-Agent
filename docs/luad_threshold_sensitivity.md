# LUAD Top-10 对疾病签名阈值的敏感性（v1）

2026-10-01。冻结排名用 RRF（k=60）融合两项：所有匹配 landmark 基因上的负 Spearman 反转（与阈值无关），以及上/下调基因集连接性（依赖阈值）。这里只重建连接性基因集：阈值网格为 FDR ∈ {0.01, 0.05, 0.10} × |log2FC| ∈ {0.58, 1.0, 1.5}；另加每个方向按 |log2FC| 取前 N 个（N = 25、50、100，FDR < 0.05）。默认设定（FDR 0.05、|log2FC| 1）精确复现冻结 Top-10。冻结排名与证据不因本分析改变。脚本：`python -m evals.luad_threshold_sensitivity`；结果：`benchmark/results/luad_threshold_sensitivity_v1.json`。糖皮质激素按名称规则判定，规则与逐个标签都写在 JSON 中；其他药物的机制取自 Broad Repurposing Hub 2025-08-18 注释。

| 设定 | 上/下调 landmark 基因 | 与冻结 Top-10 重合 | Top-10 中糖皮质激素 | 参考药平均百分位 |
|---|---:|---:|---:|---:|
| FDR<0.01, |log2FC|>=0.58 | 155/121 | 3/10 | 3 | 0.413 |
| FDR<0.01, |log2FC|>=1 | 57/50 | 10/10 | 9 | 0.362 |
| FDR<0.01, |log2FC|>=1.5 | 16/18 | 7/10 | 9 | 0.427 |
| FDR<0.05, |log2FC|>=0.58 | 155/121 | 3/10 | 3 | 0.413 |
| FDR<0.05, |log2FC|>=1 | 57/50 | 10/10 | 9 | 0.362 |
| FDR<0.05, |log2FC|>=1.5 | 16/18 | 7/10 | 9 | 0.427 |
| FDR<0.1, |log2FC|>=0.58 | 155/121 | 3/10 | 3 | 0.413 |
| FDR<0.1, |log2FC|>=1 | 57/50 | 10/10 | 9 | 0.362 |
| FDR<0.1, |log2FC|>=1.5 | 16/18 | 7/10 | 9 | 0.427 |
| top-25 per direction (FDR<0.05) | 25/25 | 8/10 | 9 | 0.386 |
| top-50 per direction (FDR<0.05) | 50/50 | 9/10 | 9 | 0.361 |
| top-100 per direction (FDR<0.05) | 100/100 | 4/10 | 4 | 0.402 |

**结论**

- **FDR 阈值不起作用**：在 961 个 landmark 基因中，|log2FC| ≥ 0.58 的基因全部已满足 FDR < 0.01，三档 FDR 给出完全相同的结果。
- **|log2FC| 阈值决定 Top-10 的面貌**：默认与更严的 1.5 下，Top-10 中有 9 个糖皮质激素；放宽到 0.58 或每方向取前 100 个基因时，与冻结 Top-10 仅重合 3–4 个，糖皮质激素降到 3–4 个。
- 放宽阈值（0.58）后的 Top-10：AZ-628（RAF inhibitor）、clocortolone-pivalate（糖皮质激素）、beclomethasone-dipropionate（糖皮质激素）、hydrocortisone（糖皮质激素）、importazole（Hub 无注释）、GDC-0941（PI3K inhibitor）、BRD-K97972722（Hub 无注释）、wortmannin（PI3K inhibitor）、NVP-BEZ235（mTOR inhibitor | PI3K inhibitor）、BRD-A47816767（Hub 无注释）。新进入者以 RAF、PI3K、PI3K/mTOR 抑制剂为主，对应 A549（KRAS 突变）中的 RAF–PI3K–mTOR 信号轴。这在生物学上可解释，但同样只是转录组假设。
- 12 种设定下始终在 Top-10 的只有：beclomethasone-dipropionate、clocortolone-pivalate、hydrocortisone。
- 参考药的平均百分位在各设定下为 0.361–0.427，始终没有明显优于随机（0.5）。

**对答辩表述的影响**：“Top-10 中 9/10 为糖皮质激素”只在预设阈值（|log2FC| ≥ 1）及更严的设定下成立，不是稳健结论。准确的说法是：糖皮质激素信号在严格签名下占主导，放宽签名后被 RAF/PI3K/mTOR 抑制剂部分取代；只有 3 个糖皮质激素在所有设定下都留在 Top-10。预设阈值在看排名之前就已固定，因此主结果不改；本分析作为稳健性披露。
