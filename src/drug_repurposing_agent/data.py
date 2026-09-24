"""Read and validate TRANSCRIPT-style expression matrices without reading labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_frame(frame: pd.DataFrame, label: str) -> None:
    if frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f"{label}: empty matrix or duplicate IDs")
    if frame.index.isna().any() or frame.columns.isna().any():
        raise ValueError(f"{label}: missing IDs")
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{label}: infinite expression values")


@dataclass(frozen=True)
class ExpressionData:
    drugs: pd.DataFrame  # genes x drugs
    diseases: pd.DataFrame  # genes x diseases

    def __post_init__(self) -> None:
        _check_frame(self.drugs, "drugs")
        _check_frame(self.diseases, "diseases")
        if set(self.drugs.index) != set(self.diseases.index):
            raise ValueError("Drug and disease gene ID sets differ")
        # A label-safe operation: align only by gene symbols, never by position.
        object.__setattr__(self, "diseases", self.diseases.loc[self.drugs.index])

    @classmethod
    def from_csv(cls, items: str | Path, users: str | Path) -> "ExpressionData":
        drugs = pd.read_csv(items, index_col=0, dtype={0: str})
        diseases = pd.read_csv(users, index_col=0, dtype={0: str})
        return cls(drugs, diseases)

    def qc(self) -> dict:
        a = self.drugs.to_numpy(dtype=float)
        b = self.diseases.to_numpy(dtype=float)
        return {
            "genes": len(self.drugs), "drugs": self.drugs.shape[1],
            "diseases": self.diseases.shape[1],
            "drug_missing": int(np.isnan(a).sum()),
            "disease_missing": int(np.isnan(b).sum()),
            "constant_drugs": int(sum(np.nanstd(a, axis=0) == 0)),
            "constant_diseases": int(sum(np.nanstd(b, axis=0) == 0)),
        }
