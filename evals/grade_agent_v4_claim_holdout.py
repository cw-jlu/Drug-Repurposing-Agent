"""Grade frozen v4 provider-visible choices against prespecified card-level gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run, verify_trace_chain
from evals.agent_v4_claim_common import DEFAULT_CASES, load_holdout, public_case
from evals.run_agent_v4_claim_holdout import extract_choice, request_payload


def grade_decision(case: dict, decision: dict | None) -> dict:
    if not isinstance(decision, dict):
        return {key: False for key in ("identity", "evidence_level", "contradiction",
                                      "action", "clinical_abstention", "citation_fidelity",
                                      "citation_coverage", "overall")}
    gold = case["gold"]
    checks = {key: decision.get(key) == gold[key]
              for key in ("identity", "evidence_level", "contradiction", "action")}
    checks["clinical_abstention"] = decision.get("clinical_use_supported") is False
    cards = case["cards"]
    reviewed = {(card["pmid"], card["scope"]) for card in cards
                if card["trust"] == "reviewed" and card["scope"] != "other_candidate"}
    cited = decision.get("citations")
    checks["citation_fidelity"] = isinstance(cited, list) and all(
        isinstance(item, dict) and (item.get("pmid"), item.get("scope")) in reviewed
        for item in cited)
    cited_pairs = (
        {(item.get("pmid"), item.get("scope")) for item in cited}
        if isinstance(cited, list) and all(isinstance(item, dict) for item in cited)
        else set()
    )
    noncontradictory = {(card["pmid"], card["scope"]) for card in cards
                        if card["trust"] == "reviewed" and card["scope"] != "other_candidate"
                        and card["kind"] != "contradiction"}
    contradictions = {(card["pmid"], card["scope"]) for card in cards
                      if card["trust"] == "reviewed" and card["kind"] == "contradiction"}
    checks["citation_coverage"] = (
        (not noncontradictory or bool(cited_pairs & noncontradictory)) and
        (not gold["contradiction"] or bool(cited_pairs & contradictions)))
    checks["overall"] = all(checks.values())
    return checks


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--choices", type=Path,
                        default=Path("artifacts/reports/agent_v4_claim_choices.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/reports/agent_v4_claim_grade.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Grade output already exists; refusing overwrite")
    holdout, signatures = load_holdout(args.cases)
    choices = json.loads(args.choices.read_text(encoding="utf-8"))
    if (choices.get("cases_sha256") != sha256_file(args.cases) or
            choices.get("status") != "prescore_choices_frozen"):
        raise ValueError("Choices are not frozen against this holdout")
    rows = choices["choices"]
    if len(rows) != len(holdout["cases"]) or [r["case_id"] for r in rows] != [c["id"] for c in holdout["cases"]]:
        raise ValueError("Frozen choice order or count differs")
    trace.emit("frozen_inputs", cases_sha256=sha256_file(args.cases),
               choices_sha256=sha256_file(args.choices), case_count=len(rows))
    grades = []
    seen_provider_paths = set()
    for case, row in zip(holdout["cases"], rows):
        provider_path = Path(row["provider_trace_file"])
        if not provider_path.is_file():
            provider_path = Path("benchmark/results/traces") / provider_path.name
        if provider_path in seen_provider_paths or sha256_file(provider_path) != row["provider_trace_sha256"]:
            raise ValueError(f"Provider trace duplicated or altered: {case['id']}")
        seen_provider_paths.add(provider_path)
        events, _ = verify_trace_chain(provider_path, require_chain=True)
        if [event["stage"] for event in events] != ["trace_started", "model_request", "model_response"]:
            raise ValueError(f"Provider request/response trace incomplete: {case['id']}")
        expected_payload = request_payload(public_case(case, signatures), choices["model"])
        if events[1]["payload"] != expected_payload:
            raise ValueError(f"Provider-visible request differs from frozen public case: {case['id']}")
        try:
            actual = extract_choice(events[2]["response"])
        except ValueError:
            actual = None
        if actual != row["decision"]:
            raise ValueError(f"Choice differs from provider-visible response: {case['id']}")
        checks = grade_decision(case, actual)
        grade = {"case_id": case["id"], "origin_rank": case["rank"],
                 "checks": checks, "provider_trace_sha256": row["provider_trace_sha256"]}
        grades.append(grade)
        trace.emit("case_graded", **grade)
    axes = list(grades[0]["checks"])
    summary = {axis: {"passed": sum(int(g["checks"][axis]) for g in grades),
                      "total": len(grades)} for axis in axes}
    result = {"status": "source_overlapping_evidence_card_holdout_not_external_biomedical_validation",
              "cases_sha256": sha256_file(args.cases),
              "choices_sha256": sha256_file(args.choices),
              "summary": summary, "grades": grades,
              "limitation": "Gold labels are author-constructed from reviewed cards and task policy; no independent clinician adjudication, prospective drug outcomes, or efficacy validation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("grade_saved", output=str(args.output), output_sha256=sha256_file(args.output),
               summary=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    traced_run("agent_v4_claim_grading", _main)
