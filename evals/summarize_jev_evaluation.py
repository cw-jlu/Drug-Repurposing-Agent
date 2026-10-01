"""Compare rules, deepseek-flash, deepseek-v4-pro and Jev on the frozen decision evals.

Reads only committed result files and writes benchmark/results/jev_evaluation_summary.json
and docs/jev_evaluation.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file

V1 = {"flash": "benchmark/results/decision_eval_v1.json",
      "v4pro": "benchmark/results/decision_eval_v1_deepseek_v4_pro.json",
      "jev": "benchmark/results/decision_eval_v1_jev.json"}
V2 = {"flash": "benchmark/results/decision_eval_v2.json",
      "v4pro": "benchmark/results/decision_eval_v2_deepseek_v4_pro.json",
      "jev": "benchmark/results/decision_eval_v2_jev.json"}
KEYS = ("accuracy", "review_recall_on_ambiguous_cases", "escalation_rate",
        "high_risk_wrong_auto_execution_rate")


def load(p: str) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> None:
    v1 = {k: load(p) for k, p in V1.items()}
    v2 = {k: load(p) for k, p in V2.items()}
    rows1 = [("J0 固定规则", v1["flash"]["layers"]["J0_fixed_rules"]),
             ("J1 flash", v1["flash"]["layers"]["J1_deepseek_structured"]),
             ("J1 v4-pro", v1["v4pro"]["layers"]["J1_deepseek_structured"]),
             ("J2 Jev（无门控）", v1["jev"]["layers"]["J2_jev"]),
             ("J3 flash + 门控", v1["flash"]["layers"]["J3_deepseek_plus_gate"]),
             ("J3 v4-pro + 门控", v1["v4pro"]["layers"]["J3_deepseek_plus_gate"]),
             ("J3 Jev + 门控", v1["jev"]["layers"]["J3_jev_gate"]),
             ("J4 Jev + 门控 + LLM 兜底", v1["jev"]["layers"]["J4_jev_gate_llm_fallback"])]
    def llm(k):
        L = v1[k]["layers"]["J1_deepseek_structured"]
        return {"brier": L.get("brier_multiclass"), "ece": (L.get("calibration_top_label") or {}).get("ece_10bin"),
                "noul_auroc": (L.get("evidence_noul") or {}).get("auroc"),
                "score_spearman": (L.get("data_quality_score_spearman_expected") or {}).get("spearman"),
                "p50_ms": L.get("latency_ms_p50"), "gate": v1[k]["layers"]["J3_deepseek_plus_gate"].get("gate_reason_counts", {})}
    calib = {"flash": llm("flash"), "v4pro": llm("v4pro"),
             "jev": {"brier": v1["jev"]["jev_calibration"]["brier_multiclass"],
                     "ece": v1["jev"]["jev_calibration"]["ece_10bin_reported_confidence"],
                     "noul_auroc": v1["jev"]["jev_calibration"]["evidence_noul"]["auroc"],
                     "score_spearman": v1["jev"]["jev_calibration"]["data_quality_score_spearman_expected"]["spearman"],
                     "p50_ms": v1["jev"]["latency_ms_p50"], "gate": v1["jev"]["gate_reason_counts"]}}
    rows2 = [("J0 旧规则", v2["flash"]["layers"]["j0_rules"]),
             ("混合 · flash 判备注", v2["flash"]["layers"]["hybrid"]),
             ("混合 · v4-pro 判备注", v2["v4pro"]["layers"]["hybrid"]),
             ("混合 · Jev 判备注", v2["jev"]["layers"]["hybrid_jev"]),
             ("混合 · Jev + 置信门控", v2["jev"]["layers"]["hybrid_jev_gated"]),
             ("完整 flash", v2["flash"]["layers"]["full_llm"]),
             ("完整 v4-pro", v2["v4pro"]["layers"]["full_llm"]),
             ("完整 Jev", v2["jev"]["layers"]["full_jev"])]
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "jev_model": v1["jev"]["model_requested"], "jev_endpoint": v1["jev"]["endpoint"],
               "v1": {name: {k: m[k] for k in KEYS} for name, m in rows1}, "v1_calibration": calib,
               "v2": {name: {"correct": m["correct"], "high_risk_wrong_auto": m["high_risk_wrong_auto_execution_rate"],
                             "concern_review_recall": m["concern_review_recall"],
                             "benign_unnecessary_review": m["benign_unnecessary_review_rate"]} for name, m in rows2},
               "sources": {p: sha256_file(Path(p)) for p in list(V1.values()) + list(V2.values())}}
    Path("benchmark/results/jev_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    f = lambda x: "—" if x is None else f"{x:.3f}"
    t1 = ["| 决策层 | 准确率 | 模糊用例转人工 | 升级率 | 高风险误执行 |", "|---|---:|---:|---:|---:|"]
    t1 += [f"| {n} | {f(m['accuracy'])} | {f(m['review_recall_on_ambiguous_cases'])} | {f(m['escalation_rate'])} | {f(m['high_risk_wrong_auto_execution_rate'])} |" for n, m in rows1]
    t2 = ["| 方案 | 正确 / 40 | 高风险误执行 | 风险备注转人工 | 无风险备注误转人工 |", "|---|---:|---:|---:|---:|"]
    t2 += [f"| {n} | {m['correct']} | {f(m['high_risk_wrong_auto_execution_rate'])} | {f(m['concern_review_recall'])} | {f(m['benign_unnecessary_review_rate'])} |" for n, m in rows2]
    c = calib
    doc = f"""# Jev 实测：决策层对比（v1 120 例 · v2 40 例）

2026-10-01。Jev 通过 OpenCode Zen 调用（`{summary['jev_endpoint']}`，模型 `{summary['jev_model']}`）。付费版 `jev-1.13` 因账户余额不足返回 HTTP 402，在得到任何答案前改用限时免费版，修订已单独提交。两组用例、策略文本和门控阈值（低风险 0.8 / 高风险 0.9）都在调用前冻结，与 LLM 层完全相同。结果：`benchmark/results/decision_eval_v1_jev.json`、`benchmark/results/decision_eval_v2_jev.json`；汇总：`benchmark/results/jev_evaluation_summary.json`（`python -m evals.summarize_jev_evaluation`）。

## v1：120 个封闭决策用例

{chr(10).join(t1)}

| 校准与速度 | flash | v4-pro | Jev |
|---|---:|---:|---:|
| 多分类 Brier（越低越好） | {f(c['flash']['brier'])} | {f(c['v4pro']['brier'])} | {f(c['jev']['brier'])} |
| ECE（10 箱，越低越好） | {f(c['flash']['ece'])} | {f(c['v4pro']['ece'])} | {f(c['jev']['ece'])} |
| 证据充分性 noul AUROC | {f(c['flash']['noul_auroc'])} | {f(c['v4pro']['noul_auroc'])} | {f(c['jev']['noul_auroc'])} |
| 数据质量评分 Spearman | {f(c['flash']['score_spearman'])} | {f(c['v4pro']['score_spearman'])} | {f(c['jev']['score_spearman'])} |
| 单次延迟 P50（ms） | {c['flash']['p50_ms']:.0f} | {c['v4pro']['p50_ms']:.0f} | {c['jev']['p50_ms']:.0f} |
| 门控拦下的低置信判断 | {c['flash']['gate'].get('low_confidence', 0)}/120 | {c['v4pro']['gate'].get('low_confidence', 0)}/120 | {c['jev']['gate'].get('low_confidence', 0)}/120 |

## v2：40 条中文风险备注

{chr(10).join(t2)}

## 结论

- **Jev 准确率最高、Brier 最低**：不加门控时准确率 {f(rows1[3][1]['accuracy'])}，高于规则与两个 DeepSeek 模型；Brier {f(c['jev']['brier'])} 明显低于 flash（{f(c['flash']['brier'])}）与 v4-pro（{f(c['v4pro']['brier'])}）。但单看校准误差 ECE，flash（{f(c['flash']['ece'])}）略好于 Jev（{f(c['jev']['ece'])}），v4-pro（{f(c['v4pro']['ece'])}）最差，所以不能说 Jev 在所有校准指标上都最好。数据质量评分的 Spearman 以 Jev 最高（{f(c['jev']['score_spearman'])}）。
- **Jev + 门控把高风险误执行降到 0**（flash + 门控为 {f(rows1[4][1]['high_risk_wrong_auto_execution_rate'])}，v4-pro 门控从不触发），代价是升级率升至 {f(rows1[6][1]['escalation_rate'])}。J4 把低置信判断交给 flash 后，准确率回到 {f(rows1[7][1]['accuracy'])}，高风险误执行 {f(rows1[7][1]['high_risk_wrong_auto_execution_rate'])}，介于两者之间。
- **混合设计对判断器不敏感**：v2 中“规则算结构化字段、模型只判备注”的混合层，无论 flash、v4-pro 还是 Jev 判备注，高风险误执行都是 0（Jev {rows2[3][1]['correct']}/40）。
- **让模型包办整个决策不稳**：完整 Jev 只有 {rows2[7][1]['correct']}/40，且出现高风险误执行；完整 v4-pro 也出现了。这再次支持“模型只做封闭的窄判断”的分工。
- 给混合层再叠加置信门控会过度保守（无风险备注 {f(rows2[4][1]['benign_unnecessary_review_rate'])} 被转人工），不建议在 v2 这类备注判断上使用。
- 延迟：Jev 单次约 {c['jev']['p50_ms']:.0f} ms，与 flash 相近，明显快于 v4-pro。免费模型，未计成本。
- 局限：用例为按成文策略生成的合成数据，每组一次运行；免费版模型可能与付费版不同。
"""
    Path("docs/jev_evaluation.md").write_text(doc, encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
