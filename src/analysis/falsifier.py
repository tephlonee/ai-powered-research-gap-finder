"""
Falsifier — the runtime evaluator, reframed as a DISCONFIRMER, not a scorer.

A gap is a claim about absence, and absence has no document to check against. So a judge that only
re-reads the candidate shares the generator's blindness and adds nothing. The useful move is to
give the judge the retrieval tool the generator leaned on least and set its goal to destruction:
"find a paper that already fills this gap." If it finds one, the gap dies (correctly). If it tries
and fails, the gap SURVIVES — and survival after a real kill-attempt is the closest thing to
validation an absence claim can get.

We fold in the subjective half of the triad (worth-knowing, tractable) as a cheap second judgment
in the same call — those ARE legitimate LLM-judge territory, unlike absence.

Retrieval here is in-corpus vector search (papers the generator didn't surface for this candidate).
Flip `use_fresh_search` on to also issue a live query — slower, higher recall.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.analysis.gap_detector import DetectedGap
from src.knowledge.vector_manager import VectorManager
from src.extraction import llm_client


class Verdict(BaseModel):
    status: str = Field(..., description="One of: open, filled, uncertain.")
    filled_by: str = Field("", description="Title/snippet of the paper that fills it, if any.")
    significance: int = Field(..., ge=1, le=5, description="How worth-knowing (1-5).")
    feasibility: int = Field(..., ge=1, le=5, description="How tractable as a project (1-5).")
    rationale: str = Field("", description="One sentence explaining the verdict.")


class FalsifiedGap(BaseModel):
    gap: DetectedGap
    verdict: Verdict

    class Config:
        arbitrary_types_allowed = True


_SYSTEM = (
    "You are a skeptical reviewer trying to DISPROVE a claimed research gap. You are given the gap "
    "and a set of retrieved papers that might already address it. Decide: does any retrieved paper "
    "already fill this gap? status='filled' if yes (name it in filled_by), 'open' if none do, "
    "'uncertain' if unclear. Then rate significance (is it worth knowing?) and feasibility (is it a "
    "tractable project?) from 1-5. Be strict: only 'filled' if a paper genuinely closes the gap."
)


def falsify(gaps: list[DetectedGap], vectors: VectorManager,
            k: int = 5) -> list[FalsifiedGap]:
    out: list[FalsifiedGap] = []
    for dg in gaps:
        query = f"{dg.statement.title}. {dg.statement.description}"
        hits = vectors.search(query, k=k)
        # exclude the gap's own supporting papers — we want what the generator didn't use
        own = set(dg.candidate.evidence_papers)
        retrieved = [h for h in hits if h["paper_id"] not in own] or hits
        retrieved_text = "\n".join(
            f"- {h.get('title') or h['paper_id']}: {h['document'][:200]}" for h in retrieved
        ) or "(no related papers retrieved)"

        prompt = (
            f"CLAIMED GAP\nTitle: {dg.statement.title}\nDescription: {dg.statement.description}\n\n"
            f"RETRIEVED PAPERS THAT MIGHT FILL IT:\n{retrieved_text}\n\n"
            "Try to disprove the gap, then rate it."
        )
        try:
            verdict = llm_client.structured(_SYSTEM, prompt, Verdict)
        except Exception:  # noqa: BLE001
            verdict = Verdict(status="uncertain", significance=3, feasibility=3,
                              rationale="falsifier unavailable")
        out.append(FalsifiedGap(gap=dg, verdict=verdict))
    return out


def survivors(falsified: list[FalsifiedGap], min_significance: int = 2) -> list[FalsifiedGap]:
    """Gaps that weren't filled and clear a significance floor, best first."""
    kept = [f for f in falsified
            if f.verdict.status != "filled" and f.verdict.significance >= min_significance]
    kept.sort(key=lambda f: (f.verdict.significance + f.verdict.feasibility,
                             f.gap.candidate.score), reverse=True)
    return kept
