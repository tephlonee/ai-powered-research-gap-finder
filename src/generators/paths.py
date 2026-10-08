"""
Swanson two-hop generator — literature-based discovery (the ABC model).

Swanson (1986) found a real medical link (fish oil <-> Raynaud's) by noticing: A relates to B,
B relates to C, but nobody had connected A and C directly. That absent shortcut is a hypothesis.

Here: over the undirected entity graph, find pairs (A, C) that share a common neighbour B but
have NO direct A-C edge. The more distinct connectors B they share, the more plausible the
missing A-C link. This is the interpretable cousin of learned link prediction — instead of a
black-box score you get a readable reason ("both connect through B1, B2, ...").

This is the generator the old code's unused get_multi_hop_paths() was meant to power.
"""
from __future__ import annotations
from itertools import combinations

import networkx as nx

from src.knowledge.graph_manager import GraphManager
from src.generators.base import GapCandidate


def _build_graph(edges: list[tuple[str, str]]) -> nx.Graph:
    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def generate(graph: GraphManager, min_shared: int = 2, max_candidates: int = 200,
             edges: list[tuple[str, str]] | None = None) -> list[GapCandidate]:
    edges = graph.all_entity_edges() if edges is None else edges
    g = _build_graph(edges)
    if g.number_of_nodes() == 0:
        return []

    # For each node B, every pair of its neighbours (A, C) with no direct edge is a candidate.
    shared: dict[tuple[str, str], set[str]] = {}
    for b in g.nodes():
        nbrs = list(g.neighbors(b))
        if len(nbrs) < 2:
            continue
        for a, c in combinations(sorted(nbrs), 2):
            if g.has_edge(a, c):
                continue
            shared.setdefault((a, c), set()).add(b)

    cands: list[GapCandidate] = []
    for (a, c), connectors in shared.items():
        if len(connectors) < min_shared:
            continue
        cands.append(
            GapCandidate(
                kind="swanson",
                gap_type="undiscovered link (ABC)",
                description=(f"'{a}' and '{c}' are never studied together, yet both connect "
                            f"through {len(connectors)} shared concept(s): "
                            f"{', '.join(sorted(connectors)[:4])}."),
                entities=[a, c],
                meta={"connectors": sorted(connectors)[:8], "n_connectors": len(connectors)},
            )
        )
    # strongest (most shared connectors) first
    cands.sort(key=lambda x: x.meta.get("n_connectors", 0), reverse=True)
    return cands[:max_candidates]
