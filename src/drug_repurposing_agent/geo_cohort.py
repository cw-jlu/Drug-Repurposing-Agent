"""Registry-driven GEO cohorts: download, verify, pair, and build a disease signature.

A disease is supported only if it has an entry in ``configs/disease_registry_v1.json``:
GEO accession, pinned file URLs and SHA-256, a sample-pairing rule, expected
counts, the value scale and the LINCS cell line used for drug signatures.
Unregistered diseases are not guessed: ``resolve_disease`` returns None and the
agent stops at manual review. Everything here is deterministic; no model is involved.
"""

from __future__ import annotations

import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import urllib.request

import numpy as np
import pandas as pd

from .differential_expression import paired_deg

REGISTRY = Path("configs/disease_registry_v1.json")
USER_AGENT = "drug-repurposing-agent/0.2 (course project; contact via GitHub cw-jlu)"


class CohortError(RuntimeError):
    pass


def load_registry(path: Path = REGISTRY) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in data["diseases"]}


def resolve_disease(question: str, registry: dict[str, dict] | None = None) -> dict | None:
    """Return the registered disease whose alias appears in the question, else None.

    Longest alias wins so that e.g. "肺腺癌" beats "肺癌" if both were registered
    to different entries; an ambiguous tie between entries returns None.
    """
    registry = load_registry() if registry is None else registry
    q = question.lower()
    hits = [(len(a), e["id"]) for e in registry.values() for a in e["aliases"] if a.lower() in q]
    if not hits:
        return None
    best = max(n for n, _ in hits)
    ids = {i for n, i in hits if n == best}
    return registry[ids.pop()] if len(ids) == 1 else None


def file_sha256(path: Path) -> str:
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files_ready(entry: dict) -> bool:
    return all(Path(f["path"]).is_file() and Path(f["path"]).stat().st_size == f["bytes"]
               for f in entry["files"].values())


def fetch(entry: dict, trace=None, opener=urllib.request.urlopen) -> dict:
    """Download missing files from the pinned URLs; every file must match its SHA-256."""
    report = {}
    for role, spec in entry["files"].items():
        path = Path(spec["path"])
        if path.is_file() and file_sha256(path) == spec["sha256"]:
            report[role] = "cached_verified"
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        request = urllib.request.Request(spec["url"], headers={"User-Agent": USER_AGENT})
        with opener(request, timeout=120) as response, part.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        digest = file_sha256(part)
        if digest != spec["sha256"]:
            part.unlink(missing_ok=True)
            raise CohortError(f"{role}: downloaded SHA-256 {digest[:12]} does not match the registry")
        part.replace(path)
        report[role] = "downloaded_verified"
        if trace is not None:
            trace.emit("geo_file_downloaded", accession=entry["accession"], role=role, url=spec["url"],
                       bytes=path.stat().st_size, sha256=digest)
    return {"accession": entry["accession"], "files": report}


def _header(path: Path, marker: str, fields: tuple[str, ...]) -> tuple[int, dict[str, list[str]]]:
    metadata = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream):
            if line.startswith(marker):
                return line_number + 1, metadata
            for field in fields:
                if line.startswith(field + "\t"):
                    metadata[field] = next(csv.reader([line], delimiter="\t"))[1:]
    raise CohortError(f"missing {marker} in {path}")


def cohort(entry: dict) -> pd.DataFrame:
    """Pair samples with the registry rule; fail if counts differ from the registry."""
    rule = entry["pairing"]
    _, meta = _header(Path(entry["files"]["series"]["path"]), "!series_matrix_table_begin",
                      ("!Sample_geo_accession", rule["field"]))
    accessions, labels = meta.get("!Sample_geo_accession", []), meta.get(rule["field"], [])
    if not accessions or len(accessions) != len(labels):
        raise CohortError("sample accessions and pairing field do not line up")
    rows = []
    for accession, label in zip(accessions, labels):
        m = re.search(rule["regex"], label)
        code = m.group(2) if m else None
        condition = {rule["case"]: "Tumor", rule["control"]: "Normal"}.get(code)
        rows.append({"sample_id": accession, "title": label, "patient_id": m.group(1) if m else None,
                     "condition": condition})
    samples = pd.DataFrame(rows)
    parsed = samples.condition.notna()
    counts = samples[parsed].groupby(["patient_id", "condition"]).size().unstack(fill_value=0)
    for col in ("Tumor", "Normal"):
        if col not in counts:
            counts[col] = 0
    complete = counts.index[(counts.Tumor == 1) & (counts.Normal == 1)]
    samples["included"] = samples.patient_id.isin(complete) & parsed
    samples["reason"] = np.where(samples.included, "complete_pair",
                                 np.where(parsed, "no_matching_pair", "unparsed_label"))
    expected = entry.get("expected", {})
    if expected.get("samples") is not None and len(samples) != expected["samples"]:
        raise CohortError(f"expected {expected['samples']} samples, GEO lists {len(samples)}")
    if expected.get("pairs") is not None and len(complete) != expected["pairs"]:
        raise CohortError(f"expected {expected['pairs']} complete pairs, found {len(complete)}")
    if len(complete) < entry.get("min_pairs", 10):
        raise CohortError(f"only {len(complete)} complete pairs")
    return samples


def _to_log2(values: pd.DataFrame, scale: str) -> tuple[pd.DataFrame, str]:
    if scale == "log2":
        return values, "log2 as deposited"
    if scale != "auto":
        raise CohortError(f"unknown value scale {scale!r}")
    # GEO2R rule for deciding whether a matrix still needs log2.
    q = np.nanquantile(values.to_numpy(dtype=float), [0, 0.25, 0.5, 0.75, 0.99, 1.0])
    needs = (q[4] > 100) or (q[5] - q[0] > 50 and q[1] > 0)
    if not needs:
        return values, "auto: already log scale (GEO2R rule)"
    return np.log2(values.where(values > 0)), "auto: log2 applied (GEO2R rule); values <= 0 set missing"


def signature(entry: dict, samples: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Paired tumour-vs-normal DEG from the raw series matrix (no cached outputs are read)."""
    series, annot = Path(entry["files"]["series"]["path"]), Path(entry["files"]["annotation"]["path"])
    start, _ = _header(series, "!series_matrix_table_begin", ())
    matrix = pd.read_csv(series, sep="\t", skiprows=start, index_col=0, compression="gzip", comment="!")
    matrix.index = matrix.index.astype(str)
    if set(samples.sample_id) - set(matrix.columns):
        raise CohortError("series matrix lacks samples listed in the metadata")
    values, scale_note = _to_log2(matrix.astype(float), entry.get("values", "auto"))
    a_start, _ = _header(annot, "!platform_table_begin", ())
    symbol_col = entry.get("annotation_symbol_column", "Gene symbol")
    annotation = (pd.read_csv(annot, sep="\t", skiprows=a_start, compression="gzip", usecols=["ID", symbol_col],
                              dtype=str, comment="!").drop_duplicates("ID").set_index("ID"))
    mapped = values.join(annotation, how="left")
    valid = mapped[symbol_col].fillna("").str.fullmatch(r"[A-Za-z][A-Za-z0-9.\-]*")
    mapped = mapped.loc[valid].copy()
    included = samples.loc[samples.included, "sample_id"].tolist()
    mapped = mapped.loc[np.isfinite(mapped[included].to_numpy(dtype=float)).all(axis=1)]
    if mapped.empty:
        raise CohortError("no probes with an unambiguous gene symbol and finite values")
    cols = list(matrix.columns)
    mapped["probe_iqr"] = mapped[cols].quantile(0.75, axis=1) - mapped[cols].quantile(0.25, axis=1)
    mapped["probe_id"] = mapped.index
    # One probe per gene, chosen without using tumour/normal labels.
    selected = (mapped.sort_values([symbol_col, "probe_iqr", "probe_id"], ascending=[True, False, True])
                .drop_duplicates(symbol_col).set_index(symbol_col))
    deg = paired_deg(selected[included], samples.loc[samples.included, ["sample_id", "patient_id", "condition"]])
    summary = {"genes": len(deg), "pairs": len(included) // 2, "value_scale": scale_note,
               "up": int((deg.included_default & (deg.direction == "up")).sum()),
               "down": int((deg.included_default & (deg.direction == "down")).sum()),
               "source": f"{series.as_posix()} (recomputed from the raw GEO series matrix)"}
    ref = entry.get("reference_signature")
    if ref:
        text = deg.to_csv(sep="\t", index=False, lineterminator="\n")
        expected = json.loads(Path(ref["manifest"]).read_text(encoding="utf-8"))["outputs"][ref["output"]]["sha256"]
        summary["matches_frozen_signature"] = expected in {
            sha256(text.encode()).hexdigest(), sha256(text.replace("\n", "\r\n").encode()).hexdigest()}
    return deg, summary
