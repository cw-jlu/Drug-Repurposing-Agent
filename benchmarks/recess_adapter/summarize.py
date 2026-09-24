"""Summarize saved repeated benchmark runs without rerunning or tuning methods."""

from pathlib import Path
import json
import sys

import numpy as np


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit("Pass one or more result JSON paths")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len({run["split"] for run in runs}) != 1:
        raise ValueError("Do not summarize different split protocols together")
    print(f"Split: {runs[0]['split']}; seeds: {', '.join(str(x['seed']) for x in runs)}")
    print("Method | Global AUC mean ± SD | Global NDCG mean ± SD")
    print("--- | ---: | ---:")
    for method in runs[0]["methods"]:
        auc = np.array([r["methods"][method]["global_AUC"] for r in runs])
        ndcg = np.array([r["methods"][method]["global_NDCG"] for r in runs])
        print(f"{method} | {auc.mean():.4f} ± {auc.std(ddof=1) if len(auc)>1 else 0:.4f} | "
              f"{ndcg.mean():.4f} ± {ndcg.std(ddof=1) if len(ndcg)>1 else 0:.4f}")


if __name__ == "__main__":
    main()
