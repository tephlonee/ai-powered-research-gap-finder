"""
Topic generator — turn surviving gaps into concrete research topics, grounded in real paper IDs.

The old version asked the LLM for "references" with no paper IDs in hand, so it hallucinated them.
Here we pass the actual supporting paper IDs/titles and require the model to phrase a researchable
question and approach around them. Every topic therefore traces back to evidence.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.analysis.falsifier import FalsifiedGap
from src.knowledge.graph_manager import GraphManager
from src.extraction import llm_client


class ResearchTopic(BaseModel):
    title: str = Field(..., description="A concise, specific research title.")
    question: str = Field(..., description="The central research question.")
    approach: str = Field(..., description="A plausible first approach/method, 2-3 sentences.")
    gap_type: str = Field(..., description="The gap type this addresses.")


class GroundedTopic(BaseModel):
    topic: ResearchTopic
    references: list[str] = Field(default_factory=list)  # real paper ids
    significance: int = 3
    feasibility: int = 3

    class Config:
        arbitrary_types_allowed = True


_SYSTEM = (
    "You are a thesis advisor. Given a validated research gap and the real papers that frame it, "
    "propose ONE concrete, researchable topic: a specific title, a central question, and a first "
    "approach. Ground it in the provided papers; do not invent citations. Be realistic about scope."
)


def generate_topics(survivors: list[FalsifiedGap], graph: GraphManager,
                    top_k: int = 10) -> list[GroundedTopic]:
    meta = graph.paper_meta()
    out: list[GroundedTopic] = []
    for f in survivors[:top_k]:
        dg = f.gap
        ref_ids = dg.candidate.evidence_papers[:6]
        titles = [f"{pid} — {meta.get(pid, {}).get('title', '')}" for pid in ref_ids]
        prompt = (
            f"VALIDATED GAP\nTitle: {dg.statement.title}\n"
            f"Description: {dg.statement.description}\nType: {dg.statement.gap_type}\n\n"
            f"FRAMING PAPERS (use as references):\n{chr(10).join(titles) or '(contextual)'}\n\n"
            "Propose one concrete research topic."
        )
        try:
            topic = llm_client.structured(_SYSTEM, prompt, ResearchTopic)
        except Exception:  # noqa: BLE001
            topic = ResearchTopic(title=dg.statement.title,
                                  question=dg.statement.description,
                                  approach="(topic generation unavailable)",
                                  gap_type=dg.statement.gap_type)
        out.append(GroundedTopic(topic=topic, references=ref_ids,
                                 significance=f.verdict.significance,
                                 feasibility=f.verdict.feasibility))
    return out
