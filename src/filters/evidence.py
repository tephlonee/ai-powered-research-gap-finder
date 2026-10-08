"""
Evidence weighting — turn "this cell is empty" into "this empty cell matters".

A candidate backed by well-established, highly-cited, recent neighbours is a real gap; the same
absence in a one-paper fringe corner is noise. We score each candidate from the metadata of its
supporting papers:

    score = log1p(total_citations) * recency_factor * log1p(paper_count)

This is the single change that upgrades the naive sparsity count into something trustworthy, and
it gives every generator's output a common ranking currency.
"""
from __future__ import annotations
import math
from datetime import datetime

from src.generators.base import GapCandidate
from src.knowledge.graph_manager import GraphManager


def _recency(years: list[int]) -> float:
    if not years:
        return 1.0
    now = datetime.now().year
    newest = max(years)
    age = max(0, now - newest)
    # gentle decay: ~1.0 for current work, ~0.6 for a decade old
    return math.exp(-age / 20.0)


def score_candidates(candidates: list[GapCandidate], graph: GraphManager) -> list[GapCandidate]:
    meta = graph.paper_meta()
    for c in candidates:
        papers = c.evidence_papers or []
        cites = sum((meta.get(pid, {}).get("citations") or 0) for pid in papers)
        years = [meta[pid]["year"] for pid in papers
                 if pid in meta and meta[pid].get("year")]
        base = math.log1p(cites) * _recency(years) * math.log1p(len(papers) + 1)
        # Swanson strength and bridge thinness carry their own signal
        if c.kind == "swanson":
            base *= 1.0 + 0.3 * c.meta.get("n_connectors", 0)
        if c.kind == "bridge":
            base *= 1.5  # interface gaps are intrinsically high-value
        if c.kind == "sparsity":
            base *= 0.5  # weak signal, explicitly down-weighted
        c.score = round(base, 4)
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates
