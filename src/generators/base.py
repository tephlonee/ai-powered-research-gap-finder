"""Shared candidate type produced by every generator and consumed by filters/detector."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class GapCandidate:
    kind: str                       # "void" | "swanson" | "bridge" | "sparsity"
    gap_type: str                   # human-facing taxonomy label (population/context, etc.)
    description: str                # short structural description (pre-LLM narration)
    entities: list[str] = field(default_factory=list)   # the entities involved
    evidence_papers: list[str] = field(default_factory=list)  # supporting paper ids
    score: float = 0.0              # filled by filters (evidence weighting etc.)
    meta: dict = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.kind}:{'|'.join(sorted(self.entities))}"
