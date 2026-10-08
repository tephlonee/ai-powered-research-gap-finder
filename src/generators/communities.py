"""
Community + bridge generator — crowded blobs vs. the interdisciplinary frontier.

Run community detection (Louvain) over the entity graph:
  * DENSE communities  = saturated subtopics (lots of internal edges) -> avoid; reported as context.
  * BRIDGES / structural holes = two sizeable communities joined by only one or two edges ->
    the interface between two mature areas that are barely studied together. These are high-value,
    low-competition thesis openings.

Implemented with networkx in-process, so no Neo4j GDS plugin is needed.
"""
from __future__ import annotations

import networkx as nx

from src.knowledge.graph_manager import GraphManager
from src.generators.base import GapCandidate


def generate(graph: GraphManager, max_bridge_edges: int = 2, min_comm_size: int = 3,
             edges: list[tuple[str, str]] | None = None
             ) -> tuple[list[GapCandidate], list[dict]]:
    edges = graph.all_entity_edges() if edges is None else edges
    g = nx.Graph()
    g.add_edges_from(edges)
    if g.number_of_nodes() < 4:
        return [], []

    try:
        communities = nx.community.louvain_communities(g, seed=42)
    except Exception:  # fallback for older networkx
        communities = list(nx.community.greedy_modularity_communities(g))

    comm_of: dict[str, int] = {}
    sizeable = []
    for i, comm in enumerate(communities):
        if len(comm) >= min_comm_size:
            sizeable.append(i)
        for n in comm:
            comm_of[n] = i

    # count inter-community edges per community pair
    inter: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for a, b in g.edges():
        ca, cb = comm_of.get(a), comm_of.get(b)
        if ca is None or cb is None or ca == cb:
            continue
        key = tuple(sorted((ca, cb)))
        inter.setdefault(key, []).append((a, b))

    bridges: list[GapCandidate] = []
    for (ca, cb), elist in inter.items():
        if ca not in sizeable or cb not in sizeable:
            continue
        if len(elist) <= max_bridge_edges:
            flat = sorted({n for e in elist for n in e})
            bridges.append(
                GapCandidate(
                    kind="bridge",
                    gap_type="interdisciplinary / interface gap",
                    description=(f"Two sizeable research clusters are joined by only "
                                f"{len(elist)} link(s) ({', '.join(flat[:4])}). The interface "
                                f"between these areas is largely unexplored."),
                    entities=flat[:6],
                    meta={"n_bridge_edges": len(elist),
                          "comm_sizes": [len(communities[ca]), len(communities[cb])]},
                )
            )

    # dense-community context (not gaps; surfaced so the detector can say "avoid these")
    dense_info = []
    for i in sizeable:
        members = list(communities[i])
        sub = g.subgraph(members)
        density = nx.density(sub)
        dense_info.append({"size": len(members),
                           "density": round(density, 3),
                           "sample": sorted(members)[:8]})
    dense_info.sort(key=lambda d: (d["density"], d["size"]), reverse=True)

    bridges.sort(key=lambda x: x.meta.get("n_bridge_edges", 99))
    return bridges, dense_info[:5]
