"""
Bipartite void generator — the core "structure finds the gap" mechanism.

Project the heterogeneous graph onto a pair of entity types and look for empty cells. Each pair
surfaces a different gap TYPE:
    Method x Component    -> population/context gap   (the roller-bearing case)
    Method x Dataset      -> benchmark/data-diversity gap
    Component x FailureMode -> coverage gap

A cell (left, right) is a candidate iff:
    * there is NO edge of the given relationship between them, AND
    * both endpoints are *active* (each already participates in at least one such edge).
The activity condition is what stops the flood of trivially-empty cells (nobody applied
bearing-fault-diagnosis to "sentiment analysis", and nobody should). Ranking/pruning of the
survivors happens in the filters stage.
"""
from __future__ import annotations

from schema.extraction_schema import RelType
from src.knowledge.graph_manager import GraphManager
from src.generators.base import GapCandidate


# (left_type, rel, right_type, gap_type_label)
_PROJECTIONS = [
    ("Method", RelType.APPLIED_TO, "Component", "population/context gap"),
    ("Method", RelType.EVALUATED_ON, "Dataset", "data-diversity / benchmark gap"),
    ("Component", RelType.HAS_FAILURE_MODE, "FailureMode", "coverage gap"),
]


def _voids_for(graph: GraphManager, left_type, rel, right_type, gap_label) -> list[GapCandidate]:
    edges = graph.edges(rel)  # [(a, b, papers)]
    present: set[tuple[str, str]] = set()
    left_active: dict[str, list[str]] = {}
    right_active: dict[str, list[str]] = {}
    for a, b, papers in edges:
        present.add((a, b))
        left_active.setdefault(a, []).extend(papers)
        right_active.setdefault(b, []).extend(papers)

    candidates: list[GapCandidate] = []
    for left in left_active:
        for right in right_active:
            if (left, right) in present:
                continue
            # evidence = papers that establish each endpoint exists/works elsewhere
            ev = sorted(set(left_active[left]) | set(right_active[right]))
            candidates.append(
                GapCandidate(
                    kind="void",
                    gap_type=gap_label,
                    description=(f"'{left}' ({left_type}) has been {rel.value.lower()} other "
                                f"{right_type.lower()}s, and '{right}' ({right_type}) is studied "
                                f"elsewhere, but no paper connects '{left}' to '{right}'."),
                    entities=[left, right],
                    evidence_papers=ev[:10],
                    meta={"left_type": left_type, "right_type": right_type, "rel": rel.value},
                )
            )
    return candidates


def generate(graph: GraphManager) -> list[GapCandidate]:
    out: list[GapCandidate] = []
    for left_type, rel, right_type, label in _PROJECTIONS:
        out += _voids_for(graph, left_type, rel, right_type, label)
    return out
