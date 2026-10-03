import gzip
from hashlib import sha256
import io
from pathlib import Path

import numpy as np
import pytest

from drug_repurposing_agent.geo_cohort import (
    CohortError, _to_log2, cohort, fetch, files_ready, load_registry, resolve_disease, signature,
)


def _entry(tmp_path, series_bytes=b"", annot_bytes=b"", **extra):
    entry = {"id": "toy", "accession": "GSE0", "aliases": ["toy disease", "玩具病"],
             "files": {"series": {"path": str(tmp_path / "s.txt.gz"), "url": "https://example.invalid/s",
                                  "bytes": len(series_bytes), "sha256": sha256(series_bytes).hexdigest()},
                       "annotation": {"path": str(tmp_path / "a.annot.gz"), "url": "https://example.invalid/a",
                                      "bytes": len(annot_bytes), "sha256": sha256(annot_bytes).hexdigest()}},
             "pairing": {"field": "!Sample_title", "regex": r"(P\d+)_([TN])$", "case": "T", "control": "N"},
             "min_pairs": 2, "values": "log2"}
    entry.update(extra)
    return entry


def _toy_files(n_pairs=4, unpaired=True, seed=0):
    rng = np.random.default_rng(seed)
    titles, ids = [], []
    for i in range(n_pairs):
        for code in "NT":
            titles.append(f"P{i}_{code}")
            ids.append(f"GSM{len(ids)}")
    if unpaired:
        titles.append("P99_T")
        ids.append(f"GSM{len(ids)}")
    lines = ["!Series_title\t\"toy\"", "!Sample_title\t" + "\t".join(f'"{t}"' for t in titles),
             "!Sample_geo_accession\t" + "\t".join(f'"{s}"' for s in ids), "!series_matrix_table_begin",
             "\"ID_REF\"\t" + "\t".join(f'"{s}"' for s in ids)]
    for g in range(30):
        base = rng.normal(8, 1, len(ids))
        shift = 3.0 if g < 5 else 0.0                       # first five probes strongly up in tumours
        vals = [b + (shift if t.endswith("_T") else 0) for b, t in zip(base, titles)]
        lines.append(f"\"PR{g}\"\t" + "\t".join(f"{v:.4f}" for v in vals))
    lines.append("!series_matrix_table_end")
    series = gzip.compress(("\n".join(lines) + "\n").encode())
    annot = gzip.compress(("!platform_table_begin\nID\tGene symbol\n"
                           + "".join(f"PR{g}\tGENE{g}\n" for g in range(30)) + "!platform_table_end\n").encode())
    return series, annot


def test_resolve_disease_uses_registry_aliases_only():
    reg = load_registry()
    assert resolve_disease("请为肺腺癌筛选候选药物", reg)["id"] == "luad_gse32863"
    assert resolve_disease("Find candidates for LUAD", reg)["id"] == "luad_gse32863"
    assert resolve_disease("请为乳腺癌筛选候选药物", reg)["id"] == "brca_gse15852"
    assert resolve_disease("请为胃癌筛选候选药物", reg) is None
    assert resolve_disease("请为肺癌筛选候选药物", reg) is None      # deliberately not an alias


def test_fetch_downloads_verifies_and_refuses_bad_hash(tmp_path):
    series, annot = _toy_files()
    entry = _entry(tmp_path, series, annot)
    served = {"https://example.invalid/s": series, "https://example.invalid/a": annot}
    opener = lambda req, timeout: io.BytesIO(served[req.full_url])
    assert fetch(entry, opener=opener)["files"] == {"series": "downloaded_verified",
                                                    "annotation": "downloaded_verified"}
    assert files_ready(entry)
    assert fetch(entry, opener=opener)["files"]["series"] == "cached_verified"
    bad = _entry(tmp_path / "x", series, annot)
    served["https://example.invalid/s"] = b"tampered"
    with pytest.raises(CohortError, match="does not match"):
        fetch(bad, opener=opener)
    assert not Path(bad["files"]["series"]["path"]).exists()


def test_cohort_pairs_samples_and_checks_expected_counts(tmp_path):
    series, annot = _toy_files()
    entry = _entry(tmp_path, series, annot)
    Path(entry["files"]["series"]["path"]).write_bytes(series)
    samples = cohort(entry)
    assert samples.included.sum() == 8 and samples.reason.tolist().count("no_matching_pair") == 1
    with pytest.raises(CohortError, match="expected 5 complete pairs"):
        cohort({**entry, "expected": {"pairs": 5}})


def test_signature_recovers_planted_up_genes(tmp_path):
    series, annot = _toy_files(n_pairs=6)
    entry = _entry(tmp_path, series, annot)
    Path(entry["files"]["series"]["path"]).write_bytes(series)
    Path(entry["files"]["annotation"]["path"]).write_bytes(annot)
    deg, summary = signature(entry, cohort(entry))
    up = set(deg.loc[deg.included_default & (deg.direction == "up"), "gene_symbol"])
    assert up == {f"GENE{g}" for g in range(5)} and summary["pairs"] == 6


def test_auto_scale_logs_unlogged_values():
    import pandas as pd
    raw = pd.DataFrame({"a": [10.0, 2000.0, 300.0], "b": [20.0, 5000.0, 0.0]})
    logged, note = _to_log2(raw, "auto")
    assert "applied" in note and np.isnan(logged.loc[2, "b"]) and logged.loc[0, "a"] == pytest.approx(np.log2(10))
    same, note = _to_log2(np.log2(raw.clip(lower=1)), "auto")
    assert "already" in note


@pytest.mark.skipif(not files_ready(load_registry()["luad_gse32863"]), reason="GSE32863 raw files not present")
def test_luad_signature_reproduces_the_frozen_manifest():
    entry = load_registry()["luad_gse32863"]
    _, summary = signature(entry, cohort(entry))
    assert (summary["up"], summary["down"], summary["matches_frozen_signature"]) == (512, 749, True)


@pytest.mark.skipif(not files_ready(load_registry()["brca_gse15852"]), reason="GSE15852 raw files not present")
def test_breast_entry_pairs_43_patients_and_reproduces_the_feasibility_signature():
    entry = load_registry()["brca_gse15852"]
    samples = cohort(entry)
    _, summary = signature(entry, samples)
    assert samples.included.sum() == 86 and (summary["up"], summary["down"]) == (115, 193)
    assert "log2 applied" in summary["value_scale"]


def test_unpaired_design_uses_case_and_control_rules(tmp_path):
    series, annot = _toy_files(n_pairs=6, unpaired=True)
    entry = _entry(tmp_path, series, annot, design="unpaired",
                   groups={"field": "!Sample_title", "case": r"_T$", "control": r"_N$"},
                   expected={"case": 7, "control": 6})
    Path(entry["files"]["series"]["path"]).write_bytes(series)
    Path(entry["files"]["annotation"]["path"]).write_bytes(annot)
    samples = cohort(entry)
    assert samples.included.sum() == 13 and set(samples.condition.dropna()) == {"Tumor", "Normal"}
    deg, summary = signature(entry, samples)
    assert (summary["design"], summary["tumor"], summary["normal"]) == ("unpaired", 7, 6)
    assert {f"GENE{g}" for g in range(5)} <= set(deg.loc[deg.included_default, "gene_symbol"])
    with pytest.raises(CohortError, match="expected 9 case"):
        cohort({**entry, "expected": {"case": 9}})
    with pytest.raises(CohortError, match="both groups"):
        cohort({**entry, "groups": {"field": "!Sample_title", "case": "P", "control": "_N$"}})


def test_paired_rule_accepts_named_groups(tmp_path):
    series, annot = _toy_files()
    entry = _entry(tmp_path, series, annot,
                   pairing={"field": "!Sample_title", "regex": r"(?P<patient>P\d+)_(?P<code>[TN])$",
                            "case": "T", "control": "N"})
    Path(entry["files"]["series"]["path"]).write_bytes(series)
    assert cohort(entry).included.sum() == 8
