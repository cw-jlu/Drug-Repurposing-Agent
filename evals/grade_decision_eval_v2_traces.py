"""Independently replay the 80 frozen v2 provider-visible requests/responses."""

from __future__ import annotations

import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run, verify_trace_chain
from evals.run_decision_eval_v1 import parse_response, request_payload
from evals.run_decision_eval_v2 import (CONFIG, POLICY, RESULT, WORK, MODEL,
                                        FROZEN_SHA256, load_frozen, note_payload,
                                        parse_note_response)


OUTPUT = Path("benchmark/results/decision_eval_v2_trace_grade.json")


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    config, policy = load_frozen()
    cases = {case["id"]: case for case in config["cases"]}
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    checkpoint = WORK / "checkpoint.jsonl"
    if result["config_sha256"] != FROZEN_SHA256 or \
            result["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("Result/checkpoint does not match frozen set")
    rows = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 80:
        raise ValueError(f"Expected 80 provider calls, found {len(rows)}")
    seen = set()
    by_key = {}
    for row in rows:
        key = (row["case_id"], row["layer"])
        if key in seen or row["case_id"] not in cases:
            raise ValueError(f"Duplicate or unknown provider record: {key}")
        seen.add(key)
        by_key[key] = row
        path = Path(row["provider_trace"])
        events, _ = verify_trace_chain(path, require_chain=True)
        if sha256_file(path) != row["provider_trace_sha256"]:
            raise ValueError(f"Provider trace hash changed: {key}")
        request = next((e["payload"] for e in events if e["stage"] == "model_request"), None)
        response = next((e["response"] for e in events if e["stage"] == "model_response"), None)
        if request is None or response is None:
            raise ValueError(f"Provider request or response missing: {key}")
        case = cases[row["case_id"]]
        expected = (note_payload(case["state"]["notes"])
                    if row["layer"] == "note_llm" else request_payload(case, policy, MODEL))
        if request != expected:
            raise ValueError(f"Provider-visible request differs from sealed source: {key}")
        encoded = json.dumps(request, ensure_ascii=False)
        if "reference_answer" in encoded or "note_is_concern" in encoded:
            raise ValueError(f"Hidden label leaked to provider: {key}")
        parsed_choice = (parse_note_response(response) if row["layer"] == "note_llm"
                         else parse_response(case, response)["choice"])
        if row["status"] != "ok" or parsed_choice != row["choice"]:
            raise ValueError(f"Provider answer/checkpoint differs: {key}")
    for scored in result["case_results"]:
        for layer in ("note_llm", "full_llm"):
            if scored["provider_trace_sha256"][layer] != by_key[(scored["case_id"], layer)]["provider_trace_sha256"]:
                raise ValueError(f"Result/provider trace crosswalk differs: {scored['case_id']}")
    grade = {"status": "all_provider_visible_payloads_and_responses_match",
             "frozen_config_sha256": sha256_file(CONFIG), "policy_sha256": sha256_file(POLICY),
             "result_sha256": sha256_file(RESULT), "checkpoint_sha256": sha256_file(checkpoint),
             "provider_records_checked": len(rows), "distinct_cases": len(cases),
             "request_matches": len(rows), "response_matches": len(rows),
             "hidden_label_leaks": 0, "trace": str(trace.path)}
    OUTPUT.write_text(json.dumps(grade, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("grade_saved", output=str(OUTPUT), output_sha256=sha256_file(OUTPUT),
               provider_records_checked=len(rows))
    print(f"Verified {len(rows)}/{len(rows)} provider traces; trace: {trace.path}")


if __name__ == "__main__":
    traced_run("decision_eval_v2_trace_grade", _main, WORK / "traces")
