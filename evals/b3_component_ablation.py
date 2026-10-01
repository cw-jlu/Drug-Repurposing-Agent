"""B3 component ablation on development seeds only (descriptive; B3 stays frozen).

Mirrors the official runner per seed: random_simple split (ptest=0.2,
random_state=seed), stanscofi random_cv_split with StratifiedKFold(5,
random_state=seed), fold model chosen by the highest fold-test AUC, then
official Lin's AUC (NS-AUC) on the validation set. Seeds: the first 10 integers
>= 1 that are not among the 100 official seeds (the same dev seeds as round 2).
The weakly correlated split is not evaluated because its single outer holdout
is the official test set. Variants: full B3, leave-one-component-out, and each
component alone.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import warnings

import numpy as np

warnings.filterwarnings("ignore")
np.asfarray = getattr(np, "asfarray", lambda v: np.asarray(v, dtype=float))

from drug_repurposing_agent.data import sha256_file  # noqa: E402
from drug_repurposing_agent.trace import TraceRecorder, traced_run  # noqa: E402

DATA = "artifacts/recess_official_b2/datasets/"


def variants() -> dict[str, tuple[str, ...]]:
    from benchmarks.recess_adapter.official_b3 import ALL_COMPONENTS
    out = {"full": ALL_COMPONENTS}
    out.update({f"-{c}": tuple(x for x in ALL_COMPONENTS if x != c) for c in ALL_COMPONENTS})
    out.update({f"only {c}": (c,) for c in ALL_COMPONENTS})
    return out


def dev_seeds(n: int = 10) -> list[int]:
    official = set(np.random.RandomState(1234).choice(range(int(1e8)), size=100).tolist())
    return [s for s in range(1, 1000) if s not in official][:n]


def run_seed(seed: int) -> dict:
    import stanscofi.datasets
    import stanscofi.training_testing as tt
    import stanscofi.utils
    from benchscofi.utils import rowwise_metrics
    from scipy.sparse import coo_array
    from sklearn.model_selection import StratifiedKFold
    from stanscofi.validation import compute_metrics
    from benchmarks.recess_adapter.official_b3 import B3, row_fusion_scores

    def lin_auc(scores, ds):
        a = rowwise_metrics.calc_auc(scores, ds, transpose=False, verbose=False)
        return float(np.mean(a)) if np.max(a) > 0 else 0.5

    def as_scores(matrix, ds):
        f = ds.folds.tocoo()
        return coo_array((matrix[f.row, f.col], (f.row, f.col)), shape=matrix.shape)

    np.random.seed(seed); random.seed(seed)
    ds = stanscofi.datasets.Dataset(**stanscofi.utils.load_dataset("TRANSCRIPT", DATA))
    (trf, vaf), _ = tt.random_simple_split(ds, 0.2, metric="euclidean", random_state=seed)
    traintest, val = ds.subset(trf), ds.subset(vaf)
    np.random.seed(seed); random.seed(seed)
    folds, _ = tt.random_cv_split(traintest, StratifiedKFold(n_splits=5, shuffle=True, random_state=seed))
    names = variants()
    test = {n: [] for n in names}; valid = {n: [] for n in names}
    for tfolds, sfolds in folds:
        tds, sds = traintest.subset(tfolds), traintest.subset(sfolds)
        model = B3(); model.fit(tds, seed=1234)
        data = model._features(sds)
        drugs = data.drugs.to_numpy(dtype=float)
        diseases = data.diseases.loc[data.drugs.index].to_numpy(dtype=float)
        for name, comps in names.items():
            matrix = row_fusion_scores(model._train_positive, drugs, diseases, comps)
            fold_scores = as_scores(matrix, sds)
            preds = model.predict(fold_scores, threshold=0)
            m, _ = compute_metrics(fold_scores, preds, sds, metrics=["AUC"], k=1, beta=1, verbose=False)
            test[name].append(float(m.loc["AUC"][m.columns[0]]))
            valid[name].append(lin_auc(as_scores(matrix, val), val))
    return {"seed": seed, "lin_auc": {n: valid[n][int(np.argmax(test[n]))] for n in names}}


def _main(trace: TraceRecorder) -> None:
    from joblib import Parallel, delayed
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--njobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/b3_component_ablation_dev.json"))
    args = parser.parse_args()
    seeds = dev_seeds(args.seeds)
    trace.emit("parameters", seeds=seeds, variants=list(variants()))
    rows = Parallel(n_jobs=args.njobs, backend="loky")(delayed(run_seed)(s) for s in seeds)
    names = list(variants())
    values = {n: np.array([r["lin_auc"][n] for r in rows]) for n in names}
    full = values["full"]
    summary = {n: {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                   "delta_vs_full": float((v - full).mean()),
                   "seeds_better_than_full": int((v > full).sum())} for n, v in values.items()}
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "split": "random_simple",
              "dev_seeds": seeds, "official_seeds_used": False, "summary": summary,
              "per_seed": rows, "b3_implementation_sha256": sha256_file(Path("benchmarks/recess_adapter/official_b3.py")),
              "trace_file": str(trace.path)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    trace.emit("saved", output=str(args.output), sha256=sha256_file(args.output))
    for n in names:
        s = summary[n]
        print(f"{n:10s} {s['mean']:.4f} ± {s['sd']:.4f}  Δ={s['delta_vs_full']:+.4f}  better={s['seeds_better_than_full']}/{len(seeds)}")


def main() -> None:
    traced_run("b3_component_ablation_dev", _main)


if __name__ == "__main__":
    main()
