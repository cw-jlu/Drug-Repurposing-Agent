"""Jev layers (J2/J3/J4) on the frozen 120-case decision eval v1.

Design fixed before any Jev call (committed with this file):
* Cases: configs/decision_eval_v1.json, verified by the frozen SHA-256 in
  evals/run_decision_eval_v1.py. Nothing in the case set, policy or gate changes.
* Jev input: `state` = {"policy": the same written policy text the LLM layer saw
  (general + node rules + risk rule), "case": the model-visible case fields}.
  Reference answers, generator metadata and note kinds are never sent.
* Questions: evals.run_decision_eval_v1.jev_questions(case) - a `choice` over the
  case options for every node, plus `score` (data quality) or `noul`
  (evidence sufficiency) where applicable.
* Layers (plan section 7.6):
    J2  Jev choice, no gate;
    J3  Jev choice through jev.gate_choice (0.8 low risk / 0.9 high risk; fail closed);
    J4  J3, but cases the gate holds back go to the frozen deepseek-flash decision
        from the v1 checkpoint, itself passed through the same gate (no new LLM call).
* Metrics: the v1 layer_metrics plus Brier/ECE of the Jev choice distribution,
  noul AUROC/Brier on clear evidence cases, expected-score Spearman on clear
  data-quality cases, latency, usage and schema failures.
Model: jev-1.13 via OpenCode Zen (OPENCODE_API_KEY). One run.

Amendment (before any scored Jev answer existed): every jev-1.13 call returned
HTTP 402 "Insufficient account funds" (49 failed calls, no answers). The run uses
the provider's limited-time free model jev-1.13-free instead; nothing else changes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from time import perf_counter

import numpy as np

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.jev import JevClient, gate_choice, validate_answer
from drug_repurposing_agent.trace import TraceRecorder, traced_run
from evals.run_decision_eval_v1 import (
    CONFIG, FROZEN_SHA256, brier_multiclass, ece, expected_score, gate_question,
    j3_decide, jev_questions, layer_metrics, load_checkpoint, load_config, noul_metrics,
    percentile, public_case, score_spearman,
)

WORK = Path("artifacts/decision_eval_jev_v1")
RESULT = Path("benchmark/results/decision_eval_v1_jev.json")
FLASH_CHECKPOINT = Path("artifacts/decision_eval_v1/checkpoint_deepseek-flash.jsonl")


def jev_state(case: dict, policy: dict) -> dict:
    node = case["node_type"]
    return {"policy": "\n\n".join((policy["general"], policy[node], policy["risk_level"])),
            "case": public_case(case)}


def layers(cases: list[dict], jev: dict[str, dict], flash: dict[str, dict]) -> dict:
    j2, j3, j4, reasons = {}, {}, {}, {}
    for case in cases:
        row = jev.get(case["id"])
        answer = row["answers"]["decision"] if row and row.get("status") == "ok" else None
        j2[case["id"]] = answer["choice"] if answer else None
        outcome = gate_choice(gate_question(case), answer, high_risk=case["risk_level"] == "high")
        reasons[case["id"]] = outcome.reason
        if outcome.action == "manual_review":
            j3[case["id"]] = case["review_option"]
            parsed = (flash.get(case["id"]) or {}).get("parsed")
            j4[case["id"]] = j3_decide(case, parsed)[0]
        else:
            j3[case["id"]] = j4[case["id"]] = outcome.action
    return {"J2_jev": j2, "J3_jev_gate": j3, "J4_jev_gate_llm_fallback": j4}, reasons


def calibration(cases: list[dict], jev: dict[str, dict]) -> dict:
    probs, refs, confs, correct, noul, exp = [], [], [], [], {}, {}
    for case in cases:
        row = jev.get(case["id"])
        if not row or row.get("status") != "ok":
            continue
        d = row["answers"]["decision"]
        probs.append(d["probabilities"]); refs.append(case["reference_answer"])
        confs.append(d["confidence"]); correct.append(d["choice"] == case["reference_answer"])
        if "noul" in row["answers"]:
            noul[case["id"]] = row["answers"]["noul"]["noul"]
        if case["node_type"] == "data_quality":
            exp[case["id"]] = expected_score(d["probabilities"])
    b, e = brier_multiclass(probs, refs), ece(confs, correct)
    return {"brier_multiclass": None if b is None else round(b, 4),
            "ece_10bin_reported_confidence": None if e is None else round(e, 4),
            "mean_reported_confidence": round(float(np.mean(confs)), 4) if confs else None,
            "evidence_noul": noul_metrics(cases, noul),
            "data_quality_score_spearman_expected": score_spearman(cases, exp)}


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="jev-1.13-free")
    parser.add_argument("--output", type=Path, default=RESULT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=150)
    args = parser.parse_args()
    config = load_config()
    cases, policy = config["cases"], config["policy"]
    trace.emit("frozen_cases_verified", config_sha256=FROZEN_SHA256, cases=len(cases), model=args.model)
    client = JevClient.from_env(args.model)
    trace.add_secret(client._api_key)
    WORK.mkdir(parents=True, exist_ok=True)
    checkpoint = WORK / f"checkpoint_{args.model}.jsonl"
    done = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r.get("config_sha256") == FROZEN_SHA256 and r.get("model") == args.model and r["status"] == "ok":
                done[r["case_id"]] = r
    todo = [c for c in cases if c["id"] not in done]
    lock, calls = threading.Lock(), {"n": 0}

    def run(case: dict) -> dict:
        questions = jev_questions(case)
        last = None
        for attempt in range(1, 4):
            with lock:
                if calls["n"] >= args.max_calls:
                    raise RuntimeError("call budget exhausted")
                calls["n"] += 1
            started = perf_counter()
            try:
                result = client.ask(jev_state(case, policy), questions)
                row = {"case_id": case["id"], "model": args.model, "status": "ok", "attempts": attempt,
                       "latency_ms": round((perf_counter() - started) * 1000, 1),
                       "answers": result["answers"], "usage": result.get("usage", {}),
                       "model_returned": result.get("model"), "provider_trace": client.last_trace_path,
                       "config_sha256": FROZEN_SHA256}
                break
            except Exception as exc:  # recorded; schema failures are kept
                last = {"case_id": case["id"], "model": args.model, "status": "error",
                        "error": type(exc).__name__, "http_status": getattr(exc, "code", None),
                        "attempts": attempt, "config_sha256": FROZEN_SHA256}
                if getattr(exc, "code", None) in (401, 402, 403):
                    break  # account/authorisation errors are not retried
                row = None
        row = row or last
        with lock:
            with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run, todo):
            done[row["case_id"]] = row
            trace.emit("case_done", case_id=row["case_id"], status=row["status"])
    flash = load_checkpoint(FLASH_CHECKPOINT, FROZEN_SHA256, "deepseek-flash")
    decisions, reasons = layers(cases, done, flash)
    ok = [r for r in done.values() if r["status"] == "ok"]
    lat = [r["latency_ms"] for r in ok]
    result = {"eval_name": "decision_eval_v1_jev", "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "config_file": str(CONFIG), "config_sha256_lf_normalized": FROZEN_SHA256,
              "model_requested": args.model, "endpoint": client.api_url, "provider": client.provider,
              "design": __doc__, "contains_api_key": False, "contains_raw_prompts": False,
              "layers": {name: layer_metrics(cases, answers) for name, answers in decisions.items()},
              "jev_calibration": calibration(cases, done),
              "gate_reason_counts": {k: list(reasons.values()).count(k) for k in sorted(set(reasons.values()))},
              "schema_or_call_failures": sum(r["status"] != "ok" for r in done.values()),
              "latency_ms_p50": percentile(lat, 50), "latency_ms_p95": percentile(lat, 95),
              "usage": {"input_tokens": sum(r["usage"].get("input_tokens", 0) for r in ok),
                        "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in ok)},
              "api_calls_this_invocation": calls["n"],
              "flash_checkpoint_sha256": sha256_file(FLASH_CHECKPOINT),
              "jev_checkpoint_sha256": sha256_file(checkpoint), "trace_file": str(trace.path)}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))
    for name, m in result["layers"].items():
        print(f"{name:26s} acc {m['accuracy']} review_recall {m['review_recall_on_ambiguous_cases']} "
              f"esc {m['escalation_rate']} hr_wrong_auto {m['high_risk_wrong_auto_execution_rate']}")
    print("calibration", result["jev_calibration"], "gate", result["gate_reason_counts"],
          "failures", result["schema_or_call_failures"], "p50", result["latency_ms_p50"])


def main() -> None:
    traced_run("decision_eval_v1_jev", _main)


if __name__ == "__main__":
    main()
