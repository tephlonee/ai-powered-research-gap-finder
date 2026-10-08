"""
Structural-analogy filter — keep the plausible missing edges, drop the absurd ones.

A bipartite void is only interesting if the two endpoints are *analogous* to things that are
already connected. "Method works on ball, roller, tapered bearings but not planetary bearings"
is a real gap, because planetary bearings sit right next to the three it already covers.
"Method works on bearings but not on sentiment analysis" is noise.

v1 measures analogy structurally (no embeddings needed): the right-hand endpoint is analogous to
the method's existing targets if it shares enough graph neighbours with them (Jaccard overlap).
This is cheap, interpretable, and graph-native. (Embedding cosine is the documented upgrade.)

Only applies to 'void' candidates; other kinds pass through untouched.
"""
from __future__ import annotations

from src.generators.base import GapCandidate
from src.knowledge.graph_manager import GraphManager


_MIN_ANALOGY = 0.10  # Jaccard overlap threshold; tune per corpus


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def filter_candidates(candidates: list[GapCandidate], graph: GraphManager) -> list[GapCandidate]:
    kept: list[GapCandidate] = []
    neighbor_cache: dict[str, set[str]] = {}

    def nbrs(name: str) -> set[str]:
        if name not in neighbor_cache:
            neighbor_cache[name] = graph.entity_neighbors(name)
        return neighbor_cache[name]

    for c in candidates:
        if c.kind != "void" or len(c.entities) != 2:
            kept.append(c)
            continue
        left, right = c.entities
        # things `left` already connects to (its established targets)
        targets = nbrs(left)
        if not targets:
            kept.append(c)   # nothing to compare against; don't over-filter
            continue
        # is `right` analogous to any established target of `left`?
        right_nbrs = nbrs(right)
        best = max((_jaccard(right_nbrs, nbrs(t)) for t in targets if t != right),
                   default=0.0)
        c.meta["analogy"] = round(best, 3)
        if best >= _MIN_ANALOGY:
            kept.append(c)
    return kept
