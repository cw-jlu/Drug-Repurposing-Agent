"""Side-by-side summary of the frozen LLM evaluations on deepseek-flash vs deepseek-v4-pro.

Reads only committed result files; both models saw identical frozen cases, prompts and
grading. Writes benchmark/results/model_replication_v4_pro.json and docs/model_replication_v4_pro.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file

FILES = {
    "planner_v3": ("benchmark/results/planner_eval_v3_deepseek_flash.json",
                   "benchmark/results/planner_eval_v3_deepseek_v4_pro.json"),
    "decision_v1": ("benchmark/results/decision_eval_v1.json",
                    "benchmark/results/decision_eval_v1_deepseek_v4_pro.json"),
    "decision_v2": ("benchmark/results/decision_eval_v2.json",
                    "benchmark/results/decision_eval_v2_deepseek_v4_pro.json"),
    "contamination_probe": ("benchmark/results/contamination_probe_v1.json",
                            "benchmark/results/contamination_probe_v1_deepseek_v4_pro.json"),
}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def planner(d: dict) -> dict:
    r = next(x for x in d["results"] if x["planner"].startswith("deepseek"))
    return {"correct": r["correct"], "cases": r["cases"], "mean_latency_ms": r["mean_latency_ms"],
            "total_tokens": r["total_tokens"]}


def decision_v1(d: dict) -> dict:
    j1, j3 = d["layers"]["J1_deepseek_structured"], d["layers"]["J3_deepseek_plus_gate"]
    return {"J1_accuracy": j1["accuracy"], "J1_high_risk_wrong_auto": j1["high_risk_wrong_auto_execution_rate"],
            "J1_review_recall_ambiguous": j1["review_recall_on_ambiguous_cases"], "J1_brier": j1.get("brier_multiclass"),
            "J1_latency_ms_p50": j1.get("latency_ms_p50"), "J3_accuracy": j3["accuracy"],
            "J3_high_risk_wrong_auto": j3["high_risk_wrong_auto_execution_rate"],
            "J3_escalation_rate": j3["escalation_rate"], "gate_reason_counts": j3.get("gate_reason_counts")}


def decision_v2(d: dict) -> dict:
    return {k: {"correct": v["correct"], "high_risk_wrong_auto": v["high_risk_wrong_auto_execution_rate"],
                "concern_review_recall": v["concern_review_recall"],
                "benign_unnecessary_review": v["benign_unnecessary_review_rate"]}
            for k, v in d["layers"].items() if k != "j0_rules"}


def probe(d: dict) -> dict:
    c = d["conditions"]
    return {"open_book_auc": c["open_book"]["positive_vs_unknown"]["auc"],
            "open_book_ci95": c["open_book"]["positive_vs_unknown"]["ci95"],
            "closed_book_auc": c["closed_book"]["positive_vs_unknown"]["auc"],
            "open_minus_closed": c["open_minus_closed_paired"]["auc_difference"],
            "open_minus_closed_ci95": c["open_minus_closed_paired"]["ci95"],
            "calls": d["call_counts"]["attempts"], "failures": d["call_counts"]["failures"]}


EXTRACT = {"planner_v3": planner, "decision_v1": decision_v1, "decision_v2": decision_v2,
           "contamination_probe": probe}


def main() -> None:
    out = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "models": ["deepseek-flash", "deepseek-v4-pro"],
           "note": "Same provider and model family; identical frozen cases, prompts and grading. Cost estimates in the source files use the flash price table and are not compared.",
           "evaluations": {}, "sources": {}}
    for name, (flash, pro) in FILES.items():
        out["evaluations"][name] = {"deepseek-flash": EXTRACT[name](load(flash)),
                                    "deepseek-v4-pro": EXTRACT[name](load(pro))}
        out["sources"][name] = {flash: sha256_file(Path(flash)), pro: sha256_file(Path(pro))}
    Path("benchmark/results/model_replication_v4_pro.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    E = out["evaluations"]
    P, V1, V2, C = E["planner_v3"], E["decision_v1"], E["decision_v2"], E["contamination_probe"]
    f, p = "deepseek-flash", "deepseek-v4-pro"
    pct = lambda x: f"{x:.3f}"
    doc = f"""# 换模型复现：deepseek-flash 对 deepseek-v4-pro

2026-10-01。把四组已冻结的 LLM 评测在 `deepseek-v4-pro` 上原样重跑：测试用例、提示词、评分规则都不变，原 flash 结果不覆盖。脚本：`python -m evals.summarize_model_replication`；汇总：`benchmark/results/model_replication_v4_pro.json`。两者属于同一厂商、同一模型系列，结论不能推广到其他厂商的模型。源文件里的成本估算都按 flash 的价格表计算，因此不做成本比较。

| 评测 | 指标 | deepseek-flash | deepseek-v4-pro |
|---|---|---:|---:|
| 规划器 v3 holdout（100 例） | 正确 | {P[f]['correct']}/100 | {P[p]['correct']}/100 |
| | 平均延迟 | {P[f]['mean_latency_ms']:.0f} ms | {P[p]['mean_latency_ms']:.0f} ms |
| 决策层 v1（120 例） | J1 准确率 | {pct(V1[f]['J1_accuracy'])} | {pct(V1[p]['J1_accuracy'])} |
| | J1 高风险误执行 | {pct(V1[f]['J1_high_risk_wrong_auto'])} | {pct(V1[p]['J1_high_risk_wrong_auto'])} |
| | J1 Brier | {pct(V1[f]['J1_brier'])} | {pct(V1[p]['J1_brier'])} |
| | J3（加门控）高风险误执行 | {pct(V1[f]['J3_high_risk_wrong_auto'])} | {pct(V1[p]['J3_high_risk_wrong_auto'])} |
| | 门控拦下的低置信判断 | {V1[f]['gate_reason_counts'].get('low_confidence', 0)}/120 | {V1[p]['gate_reason_counts'].get('low_confidence', 0)}/120 |
| 决策层 v2（40 例） | 混合层正确 / 高风险误执行 | {V2[f]['hybrid']['correct']}/40 / {pct(V2[f]['hybrid']['high_risk_wrong_auto'])} | {V2[p]['hybrid']['correct']}/40 / {pct(V2[p]['hybrid']['high_risk_wrong_auto'])} |
| | 完整 LLM 正确 / 高风险误执行 | {V2[f]['full_llm']['correct']}/40 / {pct(V2[f]['full_llm']['high_risk_wrong_auto'])} | {V2[p]['full_llm']['correct']}/40 / {pct(V2[p]['full_llm']['high_risk_wrong_auto'])} |
| 污染探针（600 对） | 开卷 AUC [95% CI] | {pct(C[f]['open_book_auc'])} [{pct(C[f]['open_book_ci95'][0])}, {pct(C[f]['open_book_ci95'][1])}] | {pct(C[p]['open_book_auc'])} [{pct(C[p]['open_book_ci95'][0])}, {pct(C[p]['open_book_ci95'][1])}] |
| | 闭卷 AUC | {pct(C[f]['closed_book_auc'])} | {pct(C[p]['closed_book_auc'])} |
| | 开卷 − 闭卷 [95% CI] | {C[f]['open_minus_closed']:+.3f} [{C[f]['open_minus_closed_ci95'][0]:+.3f}, {C[f]['open_minus_closed_ci95'][1]:+.3f}] | {C[p]['open_minus_closed']:+.3f} [{C[p]['open_minus_closed_ci95'][0]:+.3f}, {C[p]['open_minus_closed_ci95'][1]:+.3f}] |

**结论**

- **规划能力不依赖具体模型**：两者在 100 例冻结 holdout 上分别为 {P[f]['correct']} 与 {P[p]['correct']}，差距在一两例之内；v4-pro 延迟约为 flash 的 {P[p]['mean_latency_ms']/P[f]['mean_latency_ms']:.1f} 倍。
- **置信门控的阈值不能跨模型照搬**：flash 有 {V1[f]['gate_reason_counts'].get('low_confidence', 0)} 个低置信判断被门控转人工，高风险误执行从 {pct(V1[f]['J1_high_risk_wrong_auto'])} 降到 {pct(V1[f]['J3_high_risk_wrong_auto'])}；v4-pro 的 120 个判断全部高于阈值，门控一次都没触发，J3 与 J1 完全相同。两者 Brier 相近，说明 v4-pro 并没有更准，只是更自信。
- **混合设计在两个模型上都安全**：决策层 v2 中“规则算结构化字段、LLM 只判备注”的混合层，在两个模型上高风险误执行都是 0；让 LLM 包办全部决策时，v4-pro 出现 {V2[p]['full_llm']['high_risk_wrong_auto']:.0%} 的高风险误执行，flash 为 0。
- **更强的模型并没有泄漏更多**：v4-pro 开卷 AUC 为 {pct(C[p]['open_book_auc'])}，略低于 flash 的 {pct(C[f]['open_book_auc'])}，置信区间仍刚好高于 0.5；两者开卷减闭卷的差值置信区间都跨零。Strict 模式的必要性结论不变，但泄漏幅度在这份噪声标签上一直不大。
- 局限：同一厂商同一系列、每组只运行一次；其他厂商的模型需要另外的 API key。
"""
    Path("docs/model_replication_v4_pro.md").write_text(doc, encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
