"""Check necessary per-disease label counts without changing the v3 protocol."""

from __future__ import annotations

import json
from pathlib import Path

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run


SOURCE = Path("benchmark/results/external_repodb_ontology_bridge_audit.json")
OUTPUT = Path("benchmark/results/external_repodb_fold_minimum_audit.json")


def _main(trace: TraceRecorder) -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = source["rows"]
    checks = []
    for minimum in (2, 3, 5, 8):
        eligible = [row for row in rows if row["positive"] >= minimum
                    and row["failed_trial_negative"] >= minimum]
        fresh = [row for row in eligible if not row["cui_in_transcript"]]
        checks.append({"minimum_each_label": minimum,
                       "all_signatures": len(eligible),
                       "nonoverlap_signatures": len(fresh),
                       "all_signature_names": [row["signature"] for row in eligible],
                       "nonoverlap_signature_names": [row["signature"] for row in fresh]})
    result = {"status": "necessary_label_count_only_not_sufficient_for_independent_nested_cv",
              "source_sha256": sha256_file(SOURCE), "checks": checks,
              "trace": str(trace.path)}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    trace.emit("fold_minimum_audited", output=str(OUTPUT),
               output_sha256=sha256_file(OUTPUT),
               min5_all=checks[2]["all_signatures"],
               min5_nonoverlap=checks[2]["nonoverlap_signatures"])
    print(f"5-each necessary gate: {checks[2]['all_signatures']} total, "
          f"{checks[2]['nonoverlap_signatures']} non-overlapping; trace: {trace.path}")


if __name__ == "__main__":
    traced_run("repodb_fold_minimum_audit", _main, Path("artifacts/reports/traces"))
