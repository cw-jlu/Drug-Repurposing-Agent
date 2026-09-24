"""Evidence ledger and mechanical citation checks for research mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re


PMID = re.compile(r"^[1-9][0-9]{0,8}$")
DOI = re.compile(r"^10\.[0-9]{4,9}/\S+$", re.I)


@dataclass(frozen=True)
class Citation:
    kind: str
    identifier: str
    direction: str
    note: str
    evidence_type: str | None = None

    def validate(self) -> None:
        if self.kind == "PMID" and PMID.fullmatch(self.identifier):
            return
        if self.kind == "DOI" and DOI.fullmatch(self.identifier):
            return
        raise ValueError(f"Invalid citation: {self.kind} {self.identifier}")


@dataclass
class CandidateLedger:
    drug_id: str
    disease_id: str
    trace_id: str
    transcriptomic_rank: int
    reversal_scores: dict[str, float]
    drug_name: str | None = None
    perturbation_context: dict = field(default_factory=dict)
    targets: list[str] = field(default_factory=list)
    mechanism: str | None = None
    supporting_evidence: list[Citation] = field(default_factory=list)
    contradicting_evidence: list[Citation] = field(default_factory=list)
    context_evidence: list[Citation] = field(default_factory=list)
    jev_decisions: list[dict] = field(default_factory=list)
    confidence_tier: str = "insufficient_evidence"
    limitations: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.drug_id or not self.disease_id or not self.trace_id:
            raise ValueError("Ledger identifiers are required")
        if self.transcriptomic_rank < 1:
            raise ValueError("Rank must be positive")
        for citation in self.supporting_evidence + self.contradicting_evidence + self.context_evidence:
            citation.validate()
        if self.confidence_tier != "insufficient_evidence" and not self.supporting_evidence:
            raise ValueError("Evidence tier requires a supporting citation")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
