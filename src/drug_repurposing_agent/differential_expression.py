"""Paired tumor/normal differential expression for an already normalized matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    adjusted = np.full(len(p_values), np.nan)
    finite = np.flatnonzero(np.isfinite(p_values))
    if len(finite) == 0:
        return adjusted
    order = finite[np.argsort(p_values[finite])]
    ranks = np.arange(1, len(order) + 1)
    q = np.minimum.accumulate((p_values[order] * len(order) / ranks)[::-1])[::-1]
    adjusted[order] = np.clip(q, 0, 1)
    return adjusted


def paired_deg(expression: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Return per-gene tumor-minus-normal log2FC, p, and BH FDR.

    ``expression`` is gene x sample and must already contain normalized log2
    values. ``samples`` needs sample_id, patient_id, condition (Tumor/Normal).
    A missing or duplicate member of a pair is a hard error; no silent fallback
    to an unpaired comparison is allowed.
    """
    required = {"sample_id", "patient_id", "condition"}
    if not required.issubset(samples.columns):
        raise ValueError(f"Sample manifest must include {sorted(required)}")
    if expression.empty or expression.index.has_duplicates or expression.columns.has_duplicates:
        raise ValueError("Expression matrix is empty or has duplicate IDs")
    if samples["sample_id"].duplicated().any() or samples["patient_id"].isna().any():
        raise ValueError("Duplicate sample or missing patient ID")
    if set(samples["sample_id"]) != set(expression.columns):
        raise ValueError("Expression sample IDs differ from manifest")
    if set(samples["condition"]) != {"Tumor", "Normal"}:
        raise ValueError("Conditions must be Tumor and Normal")
    counts = samples.groupby(["patient_id", "condition"]).size().unstack(fill_value=0)
    if not ((counts["Tumor"] == 1) & (counts["Normal"] == 1)).all():
        raise ValueError("Every patient must have exactly one tumor and one normal sample")
    values = expression.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Paired expression matrix contains NaN or Inf")
    ids = sorted(counts.index.astype(str))
    samples = samples.assign(patient_id=samples["patient_id"].astype(str))
    tumor_ids = [samples.loc[(samples.patient_id == patient) &
                             (samples.condition == "Tumor"), "sample_id"].iloc[0] for patient in ids]
    normal_ids = [samples.loc[(samples.patient_id == patient) &
                              (samples.condition == "Normal"), "sample_id"].iloc[0] for patient in ids]
    if len(ids) < 3:
        raise ValueError("At least three complete pairs are needed")
    tumor = expression[tumor_ids].to_numpy(dtype=float)
    normal = expression[normal_ids].to_numpy(dtype=float)
    effects = (tumor - normal).mean(axis=1)
    p_values = ttest_rel(tumor, normal, axis=1).pvalue
    p_values = np.where(np.isfinite(p_values), p_values, 1.0)
    fdr = _bh_adjust(p_values)
    return pd.DataFrame({"gene_symbol": expression.index.astype(str),
                         "log2FC": effects, "p_value": p_values, "fdr": fdr,
                         "n_pairs": len(ids),
                         "direction": np.where(effects > 0, "up", np.where(effects < 0, "down", "flat")),
                         "included_default": (fdr < 0.05) & (np.abs(effects) >= 1)})
