"""Aggregate recovery statistic for the prespecified LUAD reference drugs.

Uses only measured controls (unmeasured names are excluded, never counted as
failures). Statistic: mean rank percentile among the 4,920 A549 names (lower is
better). Null: the same number of distinct ranks drawn uniformly without
replacement (exact Monte Carlo, fixed seed). Also reports the count in the top
10% and a one-sided Mann-Whitney comparison of control ranks with all ranks.

Source preference: artifacts/reports/luad_eh3226/positive_control_ranks.csv,
accepted only if its SHA-256 equals data/manifests/eh3226-luad.json. Otherwise
the ranks recorded in docs/luad_screening_report.md are used and labelled so.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from drug_repurposing_agent.data import sha256_file
from drug_repurposing_agent.trace import TraceRecorder, traced_run

N_NAMES = 4920
RANKS_CSV = Path("artifacts/reports/luad_eh3226/positive_control_ranks.csv")
DOC_RANKS = {"docetaxel": 106, "crizotinib": 241, "gefitinib": 2444, "paclitaxel": 2962,
             "erlotinib": 3147}


def load_ranks(trace: TraceRecorder) -> tuple[dict[str, int], str]:
    manifest = json.loads(Path("data/manifests/eh3226-luad.json").read_text(encoding="utf-8"))
    expected = manifest["outputs"]["positive_control_ranks.csv"]
    if RANKS_CSV.exists():
        digest = sha256_file(RANKS_CSV)
        lf = __import__("hashlib").sha256(RANKS_CSV.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if expected in {digest, lf}:
            frame = pd.read_csv(RANKS_CSV).dropna(subset=["rank"])
            trace.emit("ranks_from_regenerated_file", path=str(RANKS_CSV), sha256=digest)
            return {r.drug_name: int(r["rank"]) for _, r in frame.iterrows()}, \
                f"{RANKS_CSV} (sha256 matches data/manifests/eh3226-luad.json)"
        trace.emit("regenerated_file_hash_mismatch", path=str(RANKS_CSV), sha256=digest)
    trace.emit("ranks_from_report")
    return dict(DOC_RANKS), "docs/luad_screening_report.md (recorded ranks; EH3226 not re-run here)"


def statistics(ranks: dict[str, int], n_names: int = N_NAMES, draws: int = 200_000,
               seed: int = 20261001) -> dict:
    values = np.array(sorted(ranks.values()), dtype=float)
    k = len(values)
    observed = float((values / n_names).mean())
    rng = np.random.default_rng(seed)
    null = np.array([rng.choice(n_names, size=k, replace=False).mean() + 1 for _ in range(draws)]) / n_names
    p = float((np.sum(null <= observed) + 1) / (draws + 1))
    from scipy.stats import mannwhitneyu
    mw = mannwhitneyu(values, np.arange(1, n_names + 1), alternative="less")
    return {"measured_controls": k, "ranks": {n: ranks[n] for n in sorted(ranks, key=ranks.get)},
            "mean_percentile": observed, "null_mean_percentile": float(null.mean()),
            "permutation_p_one_sided": p, "permutation_draws": draws,
            "top10pct_count": int((values <= 0.1 * n_names).sum()),
            "top10pct_expected": round(0.1 * k, 2),
            "mann_whitney_p_one_sided": float(mw.pvalue)}


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/luad_positive_control_stats_v1.json"))
    args = parser.parse_args()
    ranks, source = load_ranks(trace)
    result = statistics(ranks)
    result.update({"generated_at": datetime.now(timezone.utc).isoformat(), "rank_source": source,
                   "n_names": N_NAMES, "prespecified_controls": 11,
                   "unmeasured_controls_excluded": 11 - result["measured_controls"],
                   "trace_file": str(trace.path)})
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))
    print(json.dumps({k: result[k] for k in ("mean_percentile", "permutation_p_one_sided",
                                             "top10pct_count", "mann_whitney_p_one_sided", "rank_source")}, indent=1))


def main() -> None:
    traced_run("luad_positive_control_stats_v1", _main)


if __name__ == "__main__":
    main()
