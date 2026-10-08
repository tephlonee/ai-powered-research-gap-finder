"""
Sparsity generator — the old code's ONLY signal, kept as ONE weak signal among many.

Entities with very few connections *may* sit in under-explored regions. But low degree can also
mean "unimportant" or "newly introduced", so on its own this is noisy — which is exactly why the
rebuild demotes it from "the gap detector" to one input that must survive the evidence/analogy
filters like any other candidate.
"""
from __future__ import annotations

from src.knowledge.graph_manager import GraphManager
from src.generators.base import GapCandidate


def generate(graph: GraphManager, max_degree: int = 2, min_degree: int = 1) -> list[GapCandidate]:
    degrees = graph.entity_degrees()
    out: list[GapCandidate] = []
    for name, deg in degrees.items():
        if min_degree <= deg <= max_degree:
            out.append(
                GapCandidate(
                    kind="sparsity",
                    gap_type="under-explored concept",
                    description=(f"'{name}' appears with only {deg} connection(s) in the corpus, "
                                f"suggesting it may be under-studied (weak signal — validate)."),
                    entities=[name],
                    meta={"degree": deg},
                )
            )
    out.sort(key=lambda x: x.meta.get("degree", 99))
    return out
