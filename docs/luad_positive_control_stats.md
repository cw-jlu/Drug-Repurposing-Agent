# LUAD 阳性对照的汇总恢复指标（v1）

2026-10-01。排名前预先冻结的 11 个 LUAD 参考药中，5 个在 EH3226 A549 的 4920 个药名里可测：docetaxel 106、crizotinib 241、gefitinib 2444、paclitaxel 2962、erlotinib 3147。其余 6 个未测得，按规则排除，不计为失败。脚本：`python -m evals.luad_positive_control_stats`；结果：`benchmark/results/luad_positive_control_stats_v1.json`。名次来源：docs/luad_screening_report.md (recorded ranks; EH3226 not re-run here)。

| 指标 | 观测值 | 随机期望 | 单侧 p |
|---|---:|---:|---:|
| 平均名次百分位（越小越好） | 0.362 | 0.500 | 0.147（无放回置换，200,000 次） |
| 进入前 10% 的个数 | 2 | 0.5 | — |
| Mann–Whitney（对照名次 < 全体名次） | — | — | 0.142 |

**结论：** 参考药整体略偏向靠前（docetaxel、crizotinib 进入前 10%），但只有 5 个可测对照，差异在统计上不显著（p ≈ 0.15）。这与 TRANSCRIPT 上“表达反转≈随机”的结论一致：本筛选**没有证明**能恢复已知 LUAD 药物。A549 为 KRAS 突变、EGFR 野生型细胞系，EGFR 抑制剂（gefitinib、erlotinib）名次靠后在生物学上可以预期，但这不能被用来事后挑选对照。
