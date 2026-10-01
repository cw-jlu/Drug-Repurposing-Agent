"""LLM knowledge-contamination probe on TRANSCRIPT (open-book vs closed-book).

Question: if an LLM sees drug/disease NAMES, can it "recover" TRANSCRIPT's known
indications from memorized knowledge? If so, any benchmark in which the model sees
identities measures label leakage, not discovery -- which is why strict mode hides them.

Stages (``python -m evals.contamination_probe <stage>``; ``all`` runs every stage):
  map      offline-first ID->name mapping, cached under artifacts/contamination_probe/
  sample   fixed-seed evaluation sample, frozen with SHA-256 BEFORE any model call
  run      open-book (names) and closed-book (reversal numbers only) model calls,
           checkpointed JSONL so reruns resume
  analyze  AUCs with bootstrap 95% CI -> benchmark/results/contamination_probe_v1.json

Only aggregate numbers are written to benchmark/results; prompts/responses stay in
ignored per-call provider traces.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import threading
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from drug_repurposing_agent.data import ExpressionData, sha256_file
from drug_repurposing_agent.deepseek import local_api_key
from drug_repurposing_agent.llm_calls import (CallStats, TracedToolCaller, ToolCallError,
                                              estimate_cost_usd, tool_payload)
from drug_repurposing_agent.ranking import score_expressions
from drug_repurposing_agent.trace import TraceRecorder, traced_run, verify_trace_chain

SEED = 20260928
POSITIVE_CAP = 300
DATA_DIR = Path("data/raw/TRANSCRIPT_dataset_v2.0.0")
WORK_DIR = Path("artifacts/contamination_probe")
RESULT_PATH = Path("benchmark/results/contamination_probe_v1.json")
SAMPLE_NAME = "sample_v1.jsonl"
CONDITIONS = ("open_book", "closed_book")
TOOL_NAME = "submit_probability"
PROBABILITY_SCHEMA = {"type": "object",
                      "properties": {"probability": {"type": "number"}},
                      "required": ["probability"], "additionalProperties": False}


# --------------------------------------------------------------------------- mapping

def _http_json(url: str, data: dict | None = None, timeout: float = 60) -> object:
    body = urlencode(data).encode("utf-8") if data is not None else None
    request = Request(url, data=body, headers={"User-Agent": "DrugRepurposingAgent/0.1"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception:  # public API hiccup: bounded retry
            if attempt == 3:
                raise
            sleep(2 * (2 ** attempt))
    raise RuntimeError("unreachable")


def _clean_name(value: object) -> str | None:
    if isinstance(value, list):
        value = next((item for item in value if isinstance(item, str) and item.strip()), None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def fetch_drugbank_names(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for start in range(0, len(ids), 200):
        chunk = ids[start:start + 200]
        hits = _http_json("https://mychem.info/v1/query",
                          {"q": ",".join(chunk), "scopes": "drugbank.id",
                           "fields": "drugbank.name,chembl.pref_name"})
        for hit in hits:
            query = hit.get("query")
            if query in out or hit.get("notfound"):
                continue
            drugbank = hit.get("drugbank")
            if isinstance(drugbank, list):
                drugbank = drugbank[0] if drugbank else {}
            name = _clean_name((drugbank or {}).get("name"))
            source = "mychem.info:drugbank.name"
            if not name:
                chembl = hit.get("chembl")
                if isinstance(chembl, list):
                    chembl = chembl[0] if chembl else {}
                name = _clean_name((chembl or {}).get("pref_name"))
                name = name.title() if name else None
                source = "mychem.info:chembl.pref_name"
            if name:
                out[query] = {"name": name, "source": source}
    return out


def fetch_pubchem_names(ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    cids = [identifier[3:] for identifier in ids]
    if not cids:
        return out
    data = _http_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                      + ",".join(cids) + "/property/Title/JSON")
    for row in data["PropertyTable"]["Properties"]:
        if row.get("Title"):
            out[f"CID{row['CID']}"] = {"name": row["Title"], "source": "pubchem:Title"}
    return out


def fetch_medgen_names(cuis: list[str]) -> dict[str, dict]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    out: dict[str, dict] = {}
    for start in range(0, len(cuis), 40):
        chunk = cuis[start:start + 40]
        term = " OR ".join(chunk)
        search = _http_json(base + "esearch.fcgi?" + urlencode(
            {"db": "medgen", "term": term, "retmax": 500, "retmode": "json",
             "tool": "DrugRepurposingAgent"}))
        uids = search["esearchresult"]["idlist"]
        sleep(0.4)
        if not uids:
            continue
        summary = _http_json(base + "esummary.fcgi?" + urlencode(
            {"db": "medgen", "id": ",".join(uids), "retmode": "json",
             "tool": "DrugRepurposingAgent"}))["result"]
        sleep(0.4)
        wanted = set(chunk)
        for uid in summary.get("uids", []):
            row = summary[uid]
            cui = row.get("conceptid")
            if cui in wanted and cui not in out and row.get("title"):
                out[cui] = {"name": row["title"], "source": "ncbi_medgen:esummary.title"}
    return out


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "ratings_mat.csv", index_col=0)


def run_map(trace: TraceRecorder, work: Path = WORK_DIR, data_dir: Path = DATA_DIR) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    ratings = load_ratings(data_dir)
    drug_ids = [str(x) for x in ratings.index]
    disease_ids = [str(x) for x in ratings.columns]
    drug_path, disease_path = work / "drug_names.json", work / "disease_names.json"
    offline_note = ("Repurposing Hub files carry Broad IDs, names, InChIKeys and PubChem CIDs "
                    "but no DrugBank IDs or MedGen CUIs; no in-repo DrugBank/MedGen crosswalk "
                    "exists, so public APIs were used.")
    if drug_path.exists():
        drugs = json.loads(drug_path.read_text(encoding="utf-8"))
    else:
        db_ids = [x for x in drug_ids if x.startswith("DB")]
        cid_ids = [x for x in drug_ids if x.startswith("CID")]
        drugs = fetch_drugbank_names(db_ids)
        drugs.update(fetch_pubchem_names(cid_ids))
        drug_path.write_text(json.dumps(drugs, indent=1, ensure_ascii=False), encoding="utf-8")
    if disease_path.exists():
        diseases = json.loads(disease_path.read_text(encoding="utf-8"))
    else:
        diseases = fetch_medgen_names(disease_ids)
        disease_path.write_text(json.dumps(diseases, indent=1, ensure_ascii=False),
                                encoding="utf-8")
    values = ratings.to_numpy()
    mapped_rows = np.array([x in drugs for x in drug_ids])
    mapped_cols = np.array([x in diseases for x in disease_ids])
    both = mapped_rows[:, None] & mapped_cols[None, :]
    coverage = {
        "drugs_total": len(drug_ids), "drugs_mapped": int(mapped_rows.sum()),
        "drugbank_ids": sum(x.startswith("DB") for x in drug_ids),
        "pubchem_cids": sum(x.startswith("CID") for x in drug_ids),
        "diseases_total": len(disease_ids), "diseases_mapped": int(mapped_cols.sum()),
        "positives_total": int((values == 1).sum()),
        "positives_both_mapped": int(((values == 1) & both).sum()),
        "negatives_total": int((values == -1).sum()),
        "negatives_both_mapped": int(((values == -1) & both).sum()),
        "unknown_both_mapped": int(((values == 0) & both).sum()),
        "unmapped_drug_ids": [x for x in drug_ids if x not in drugs],
        "unmapped_disease_ids": [x for x in disease_ids if x not in diseases],
        "sources": sorted({v["source"] for v in list(drugs.values()) + list(diseases.values())}),
        "offline_first_note": offline_note,
    }
    (work / "mapping_coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    trace.emit("mapping_done", **{k: v for k, v in coverage.items()
                                  if not k.startswith("unmapped")})
    return coverage


# --------------------------------------------------------------------------- sampling

def reversal_features(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-free Spearman reversal score and its per-drug rank percentile (0..1, 1=best)."""
    data = ExpressionData.from_csv(data_dir / "items.csv", data_dir / "users.csv")
    score = score_expressions(data)["spearman_reversal"]
    pct = score.rank(axis=1, method="average").sub(1).div(score.shape[1] - 1)
    return score, pct


def build_sample(ratings: pd.DataFrame, drug_names: dict, disease_names: dict,
                 score: pd.DataFrame | None = None, pct: pd.DataFrame | None = None,
                 seed: int = SEED, positive_cap: int = POSITIVE_CAP) -> list[dict]:
    """All mapped positives (capped), equal-size random mapped unknowns, all mapped -1."""
    pairs = {1: [], 0: [], -1: []}
    for drug in sorted(map(str, ratings.index)):
        if drug not in drug_names:
            continue
        row = ratings.loc[drug]
        for disease in sorted(map(str, ratings.columns)):
            if disease in disease_names:
                pairs[int(row[disease])].append((drug, disease))
    rng = np.random.default_rng(seed)
    positives = pairs[1]
    if len(positives) > positive_cap:
        idx = np.sort(rng.choice(len(positives), positive_cap, replace=False))
        positives = [positives[i] for i in idx]
    idx = np.sort(rng.choice(len(pairs[0]), len(positives), replace=False))
    unknown = [pairs[0][i] for i in idx]
    chosen = ([(d, s, 1) for d, s in positives] + [(d, s, 0) for d, s in unknown]
              + [(d, s, -1) for d, s in pairs[-1]])
    order = rng.permutation(len(chosen))
    rows = []
    for number, i in enumerate(order, start=1):
        drug, disease, label = chosen[i]
        row = {"pair_id": f"P{number:04d}", "drug_id": drug, "disease_id": disease,
               "drug_name": drug_names[drug]["name"], "disease_name": disease_names[disease]["name"],
               "label": label}
        if score is not None and pct is not None:
            row["reversal_score"] = round(float(score.loc[drug, disease]), 6)
            row["reversal_drug_percentile"] = round(float(pct.loc[drug, disease]), 4)
        rows.append(row)
    return rows


def canonical_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                   for row in rows).encode("utf-8")


def freeze_sample(rows: list[dict], path: Path) -> str:
    """Write the sample once; refuse to overwrite a differing frozen file."""
    data = canonical_bytes(rows)
    digest = hashlib.sha256(data).hexdigest()
    hash_path = path.with_suffix(".sha256")
    if path.exists():
        existing = hashlib.sha256(path.read_bytes()).hexdigest()
        if existing != digest:
            raise ValueError("Frozen sample exists with a different hash; refusing to resample")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    hash_path.write_text(digest + "\n", encoding="utf-8")
    return digest


def load_frozen_sample(path: Path) -> tuple[list[dict], str]:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    if digest != expected:
        raise ValueError("Frozen sample hash mismatch")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return rows, digest


def run_sample(trace: TraceRecorder, work: Path = WORK_DIR, data_dir: Path = DATA_DIR) -> dict:
    drugs = json.loads((work / "drug_names.json").read_text(encoding="utf-8"))
    diseases = json.loads((work / "disease_names.json").read_text(encoding="utf-8"))
    score, pct = reversal_features(data_dir)
    rows = build_sample(load_ratings(data_dir), drugs, diseases, score, pct)
    digest = freeze_sample(rows, work / SAMPLE_NAME)
    counts = {str(k): sum(r["label"] == k for r in rows) for k in (1, 0, -1)}
    trace.emit("sample_frozen", sample_sha256=digest, seed=SEED, counts=counts,
               path=str(work / SAMPLE_NAME))
    return {"sample_sha256": digest, "counts": counts}


# --------------------------------------------------------------------------- prompts

OPEN_SYSTEM = ("You are a pharmacology knowledge probe for a research benchmark audit. "
               "Answer from your background knowledge only. Call submit_probability exactly "
               "once with a calibrated probability between 0 and 1.")
CLOSED_SYSTEM = ("You are a calibrated estimator for a research benchmark audit. You see only "
                 "anonymized numeric features of one drug-disease pair; no identities are given "
                 "and none should be guessed. Call submit_probability exactly once with a "
                 "probability between 0 and 1.")


def open_book_payload(row: dict, model: str) -> dict:
    user = (f'Is "{row["drug_name"]}" an approved or established treatment for '
            f'"{row["disease_name"]}"? Answer with a probability from 0 to 1.')
    payload = tool_payload(model, OPEN_SYSTEM, user, TOOL_NAME,
                           "Probability that the drug is an approved/established treatment.",
                           PROBABILITY_SCHEMA, max_tokens=60)
    payload["temperature"] = 0
    return payload


def closed_book_payload(row: dict, model: str) -> dict:
    user = json.dumps({
        "question": ("What is the probability that this drug is an approved or established "
                     "treatment for this disease? Answer with a probability from 0 to 1."),
        "features": {
            "spearman_reversal_score": row["reversal_score"],
            "spearman_reversal_score_meaning": (
                "negative Spearman correlation between the drug-induced and the disease "
                "gene-expression signatures over 12,096 genes; range -1..1, higher means the "
                "drug more strongly reverses the disease signature"),
            "per_drug_rank_percentile": row["reversal_drug_percentile"],
            "per_drug_rank_percentile_meaning": (
                "percentile of this disease among 151 diseases ranked by this drug's reversal "
                "score; 1.0 = the disease this drug reverses most strongly"),
        },
        "base_rate_hint": "about 0.4% of all drug-disease pairs in the source matrix are known treatments",
    }, ensure_ascii=False)
    payload = tool_payload(model, CLOSED_SYSTEM, user, TOOL_NAME,
                           "Probability that the drug is an approved/established treatment.",
                           PROBABILITY_SCHEMA, max_tokens=60)
    payload["temperature"] = 0
    return payload


def validate_probability(arguments: dict) -> dict:
    value = arguments.get("probability")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("probability must be numeric")
    if not 0.0 <= float(value) <= 1.0 or not np.isfinite(value):
        raise ValueError("probability outside [0, 1]")
    return {"probability": float(value)}


def assert_closed_book_blind(payload: dict, row: dict) -> None:
    """The closed-book request must never contain identities or labels."""
    text = json.dumps(payload["messages"][1]["content"], ensure_ascii=False).lower()
    for key in ("drug_id", "disease_id", "drug_name", "disease_name", "pair_id"):
        value = str(row[key]).lower().strip()
        if len(value) >= 4 and re.search(r"(?<![a-z0-9])" + re.escape(value) + r"(?![a-z0-9])",
                                         text):
            raise ValueError(f"closed-book payload leaks {key}")
    if '"label"' in text:
        raise ValueError("closed-book payload leaks label")


# --------------------------------------------------------------------------- metrics

def auc(positive: list[float] | np.ndarray, negative: list[float] | np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as 0.5."""
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC needs both classes")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def bootstrap_auc_ci(positive, negative, n_boot: int = 2000, seed: int = SEED,
                     alpha: float = 0.05) -> dict:
    """Stratified percentile bootstrap (resample each class with replacement)."""
    pos = np.asarray(positive, dtype=float)
    neg = np.asarray(negative, dtype=float)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = auc(pos[rng.integers(0, len(pos), len(pos))],
                       neg[rng.integers(0, len(neg), len(neg))])
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {"auc": round(auc(pos, neg), 4), "ci95": [round(float(low), 4), round(float(high), 4)],
            "n_positive": int(len(pos)), "n_negative": int(len(neg)), "n_bootstrap": n_boot,
            "bootstrap_seed": seed}


def paired_auc_difference(labels, first, second, n_boot: int = 2000, seed: int = SEED) -> dict:
    """AUC(first) - AUC(second) on the same pairs; stratified paired bootstrap CI."""
    labels = np.asarray(labels)
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    point = auc(first[pos], first[neg]) - auc(second[pos], second[neg])
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        p, q = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        stats[b] = auc(first[p], first[q]) - auc(second[p], second[q])
    low, high = np.quantile(stats, [0.025, 0.975])
    return {"auc_difference": round(float(point), 4),
            "ci95": [round(float(low), 4), round(float(high), 4)],
            "n_positive": int(len(pos)), "n_negative": int(len(neg)), "n_bootstrap": n_boot}


# --------------------------------------------------------------------------- run

def load_checkpoint(path: Path, sample_sha: str) -> dict[tuple[str, str], dict]:
    done: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["sample_sha256"] != sample_sha:
            raise ValueError("Checkpoint belongs to a different frozen sample")
        if row.get("status") == "ok":
            done[(row["pair_id"], row["condition"])] = row
    return done


def run_calls(trace: TraceRecorder, caller: TracedToolCaller, rows: list[dict], sample_sha: str,
              checkpoint: Path, model: str, workers: int = 8, max_calls: int = 1400) -> dict:
    done = load_checkpoint(checkpoint, sample_sha)
    todo = [(row, condition) for condition in CONDITIONS for row in rows
            if (row["pair_id"], condition) not in done]
    trace.emit("run_plan", already_done=len(done), todo=len(todo), workers=workers,
               max_calls=max_calls, sample_sha256=sample_sha)
    if len(todo) > max_calls:
        raise ValueError(f"{len(todo)} pending calls exceed the {max_calls}-call budget")
    lock = threading.Lock()
    failures = 0

    def one(row: dict, condition: str) -> dict:
        if condition == "open_book":
            payload = open_book_payload(row, model)
        else:
            payload = closed_book_payload(row, model)
            assert_closed_book_blind(payload, row)
        try:
            result = caller.call(payload, TOOL_NAME, validate_probability)
            return {"pair_id": row["pair_id"], "condition": condition, "status": "ok",
                    "probability": result["arguments"]["probability"],
                    "attempts": result["attempts"], "usage": result["usage"],
                    "model": result["model"],
                    "provider_trace_files": result["provider_trace_files"],
                    "sample_sha256": sample_sha}
        except ToolCallError as exc:
            return {"pair_id": row["pair_id"], "condition": condition, "status": "failed",
                    "error": str(exc), "sample_sha256": sample_sha}

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
        futures = [pool.submit(one, row, condition) for row, condition in todo]
        for count, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            with lock:
                with checkpoint.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if record["status"] != "ok":
                    failures += 1
                    trace.emit("pair_failed", pair_id=record["pair_id"],
                               condition=record["condition"], error=record["error"])
            if count % 100 == 0:
                trace.emit("progress", completed=count, pending=len(todo) - count,
                           stats=caller.stats.as_dict())
                print(f"{count}/{len(todo)} calls", flush=True)
    stats = caller.stats.as_dict()
    trace.emit("calls_finished", stats=stats, failures=failures)
    return stats


# --------------------------------------------------------------------------- analyze

def analyze(rows: list[dict], records: dict[tuple[str, str], dict], sample_sha: str,
            model: str, coverage: dict, call_stats: dict, trace_path: str,
            checkpoint_sha: str | None) -> dict:
    by_label = {k: [r for r in rows if r["label"] == k] for k in (1, 0, -1)}
    conditions = {}
    for condition in CONDITIONS:
        values = {k: [records[(r["pair_id"], condition)]["probability"] for r in group
                      if (r["pair_id"], condition) in records] for k, group in by_label.items()}
        entry = {"positive_vs_unknown": bootstrap_auc_ci(values[1], values[0]),
                 "answered": {str(k): len(v) for k, v in values.items()},
                 "mean_probability": {str(k): round(float(np.mean(v)), 4) if v else None
                                      for k, v in values.items()},
                 "median_probability": {str(k): round(float(np.median(v)), 4) if v else None
                                        for k, v in values.items()},
                 "count_probability_ge_0_5": {str(k): int(sum(x >= 0.5 for x in v))
                                              for k, v in values.items()},
                 "distinct_probability_values": len({x for v in values.values() for x in v})}
        if values[-1]:
            entry["positive_vs_known_negative_descriptive"] = bootstrap_auc_ci(values[1], values[-1])
        conditions[condition] = entry
    raw = {k: [r["reversal_score"] for r in group] for k, group in by_label.items()}
    conditions["raw_reversal_score"] = {
        "positive_vs_unknown": bootstrap_auc_ci(raw[1], raw[0]),
        "positive_vs_known_negative_descriptive": bootstrap_auc_ci(raw[1], raw[-1]) if raw[-1] else None,
        "mean_score": {str(k): round(float(np.mean(v)), 4) for k, v in raw.items() if v}}
    paired = [r for r in by_label[1] + by_label[0]
              if all((r["pair_id"], c) in records for c in CONDITIONS)]
    if paired and any(r["label"] == 1 for r in paired) and any(r["label"] == 0 for r in paired):
        conditions["open_minus_closed_paired"] = paired_auc_difference(
            [r["label"] for r in paired],
            [records[(r["pair_id"], "open_book")]["probability"] for r in paired],
            [records[(r["pair_id"], "closed_book")]["probability"] for r in paired])
    usage = call_stats.get("usage", {})
    return {
        "experiment": "contamination_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "aggregate_only_no_raw_prompts",
        "dataset": "TRANSCRIPT v2.0.0 ratings_mat.csv / items.csv / users.csv",
        "model": model,
        "sample": {"sha256": sample_sha, "seed": SEED, "positive_cap": POSITIVE_CAP,
                   "counts": {str(k): len(v) for k, v in by_label.items()},
                   "frozen_before_model_calls": True},
        "mapping_coverage": {k: v for k, v in coverage.items() if not k.startswith("unmapped")},
        "conditions": conditions,
        "primary_metric": "AUC positives (1) vs unknown (0); -1 comparison is descriptive (tiny n)",
        "call_counts": call_stats,
        "cost_estimate": estimate_cost_usd(usage),
        "trace_file": trace_path,
        "checkpoint_sha256": checkpoint_sha,
        "interpretation": ("High open-book AUC indicates name-based recall of memorized indications "
                           "(label leakage), not discovery; closed-book and raw reversal measure "
                           "what the expression signal alone supports."),
    }


def _main(trace: TraceRecorder) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["map", "sample", "run", "analyze", "all"])
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-calls", type=int, default=1400)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DRUG_AGENT_TRACE_DIR", str(work / "provider_traces"))
    trace.emit("probe_started", probe_stage=args.stage, model=args.model)
    coverage = None
    if args.stage in ("map", "all"):
        coverage = run_map(trace, work)
        print(json.dumps({k: v for k, v in coverage.items() if not k.startswith("unmapped")},
                         indent=1))
    if args.stage in ("sample", "all"):
        print(json.dumps(run_sample(trace, work), indent=1))
    sample_path = work / SAMPLE_NAME
    checkpoint = work / "checkpoint_v1.jsonl"
    stats_path = work / "call_stats.json"
    if args.stage in ("run", "all"):
        rows, sample_sha = load_frozen_sample(sample_path)
        trace.emit("frozen_sample_verified", sample_sha256=sample_sha)
        key = local_api_key()
        trace.add_secret(key)
        previous = json.loads(stats_path.read_text()) if stats_path.exists() else None
        caller = TracedToolCaller(key, model=args.model, stats=CallStats())
        stats = run_calls(trace, caller, rows, sample_sha, checkpoint, args.model,
                          args.workers, args.max_calls)
        if previous:  # accumulate across resumed runs
            stats = {"attempts": stats["attempts"] + previous["attempts"],
                     "successes": stats["successes"] + previous["successes"],
                     "failures": stats["failures"] + previous["failures"],
                     "usage": {k: stats["usage"].get(k, 0) + previous["usage"].get(k, 0)
                               for k in set(stats["usage"]) | set(previous["usage"])}}
        stats_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    if args.stage in ("analyze", "all"):
        rows, sample_sha = load_frozen_sample(sample_path)
        records = load_checkpoint(checkpoint, sample_sha)
        sampled = [path for rec in list(records.values())[:20]
                   for path in rec.get("provider_trace_files", [])]
        for path in sampled:  # spot-check provider traces are sealed
            verify_trace_chain(Path(path), require_chain=True)
        coverage = coverage or json.loads((work / "mapping_coverage.json").read_text())
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        result = analyze(rows, records, sample_sha, args.model, coverage, stats,
                         str(trace.path), sha256_file(checkpoint) if checkpoint.exists() else None)
        result["provider_traces_spot_checked"] = len(sampled)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        trace.emit("results_saved", output=str(args.output), output_sha256=sha256_file(args.output),
                   conditions={k: v["positive_vs_unknown"] for k, v in result["conditions"].items() if "positive_vs_unknown" in v})
        print(json.dumps({k: v["positive_vs_unknown"] for k, v in result["conditions"].items()
                          if "positive_vs_unknown" in v},
                         indent=1))


if __name__ == "__main__":
    traced_run("contamination_probe", _main, WORK_DIR / "traces")
