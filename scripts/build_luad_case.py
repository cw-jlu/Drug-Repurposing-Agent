"""Package the frozen LUAD screen with validation, trace, and cost record."""

import argparse
import json
from pathlib import Path

from drug_repurposing_agent.jev import JevClient
from drug_repurposing_agent.luad_case import build_luad_case


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-jev", action="store_true",
                        help="Explicitly send a narrow research-routing question to TypeSafe Jev")
    args = parser.parse_args()
    report = build_luad_case(
        Path("artifacts/reports/luad_eh3226"),
        Path("data/manifests/gse32863.json"),
        Path("data/manifests/eh3226-luad.json"),
        Path("artifacts/reports/luad_case"),
        jev_client=JevClient.from_env() if args.use_jev else None,
    )
    print(json.dumps({"status": report["status"], "run_id": report["run_id"],
                      "candidates": len(report["candidates"]),
                      "measured_controls": sum(x["rank"] is not None for x in
                                               report["prespecified_controls"]),
                      "output": "artifacts/reports/luad_case/case_report.json"}, indent=2))
