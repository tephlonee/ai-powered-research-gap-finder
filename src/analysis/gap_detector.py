"""
Gap detector — the LLM NARRATES structural candidates; it does not invent gaps.

This is the inversion the whole rebuild is about. The old detect_gaps dumped a few abstracts into
GPT and asked it to free-associate gaps. Here the graph has already produced ranked, evidence-
backed candidates; the model's only job is to turn each into a clear, well-phrased gap statement,
keeping the provenance attached. Cheap, grounded, and traceable.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.generators.base import GapCandidate
from src.knowledge.graph_manager import GraphManager
from src.extraction import llm_client


class GapStatement(BaseModel):
    title: str = Field(..., description="One-line name for the gap.")
    description: str = Field(..., description="2-4 sentences explaining the gap and why it exists.")
    gap_type: str = Field(..., description="Taxonomy label, copied from the candidate.")


_SYSTEM = (
    "You are a research advisor. You are given a STRUCTURAL gap candidate found by analysing a "
    "knowledge graph of a research field — it is grounded in real papers. Your job is ONLY to "
    "phrase it as a clear research gap: do not invent facts beyond the candidate, do not add gaps. "
    "Be precise and concrete. Keep the gap_type exactly as given."
)


class DetectedGap(BaseModel):
    statement: GapStatement
    candidate: GapCandidate

    class Config:
        arbitrary_types_allowed = True


def narrate(candidates: list[GapCandidate], graph: GraphManager,
            top_k: int = 15) -> list[DetectedGap]:
    meta = graph.paper_meta()
    out: list[DetectedGap] = []
    for c in candidates[:top_k]:
        titles = [meta.get(pid, {}).get("title", pid) for pid in c.evidence_papers[:5]]
        prompt = (
            f"Gap type: {c.gap_type}\n"
            f"Structural finding: {c.description}\n"
            f"Entities involved: {', '.join(c.entities)}\n"
            f"Supporting papers: {'; '.join(t for t in titles if t) or '(contextual)'}\n\n"
            "Phrase this as a research gap."
        )
        try:
            stmt = llm_client.structured(_SYSTEM, prompt, GapStatement)
        except Exception as exc:  # noqa: BLE001
            stmt = GapStatement(title=c.description[:80], description=c.description,
                                gap_type=c.gap_type)
        out.append(DetectedGap(statement=stmt, candidate=c))
    return out
