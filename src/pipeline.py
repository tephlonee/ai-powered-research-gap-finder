"""
Pipeline — wires the whole gap-finding flow and embodies the guiding inversion:

    plan -> ingest -> extract -> verify -> canonicalize -> build(graph + vectors)
         -> GENERATE candidates (bipartite voids, Swanson paths, bridges, sparsity)
         -> FILTER (evidence weighting, structural analogy)
         -> NARRATE (LLM phrases the structural candidates)
         -> FALSIFY (disconfirm + significance/feasibility)
         -> TOPICS (grounded in real paper IDs)

Structure finds the candidates; filters prune them; the LLM only narrates and the falsifier
validates. Nothing is free-associated.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from tqdm import tqdm

from config import config
from src.cache.cache import Cache
from src.ingestion import query_planner, paper_fetcher
from src.extraction import extractor, verifier
from src.canonicalization.resolver import Resolver
from src.knowledge.graph_manager import GraphManager
from src.knowledge.vector_manager import VectorManager
from schema.extraction_schema import EntityType, RelType
from src.generators import bipartite, paths, communities, sparsity
from src.filters import evidence, analogy
from src.analysis import gap_detector, falsifier, topic_generator


@dataclass
class PipelineResult:
    topic: str
    n_papers: int = 0
    n_entities: int = 0
    dense_clusters: list = field(default_factory=list)
    topics: list = field(default_factory=list)   # GroundedTopic
    survivors: list = field(default_factory=list)  # FalsifiedGap


def _type_lists(extraction):
    """Map extraction list-fields to (EntityType, names)."""
    return [
        (EntityType.TASK, extraction.tasks),
        (EntityType.METHOD, extraction.methods),
        (EntityType.COMPONENT, extraction.components),
        (EntityType.DATASET, extraction.datasets),
        (EntityType.METRIC, extraction.metrics),
        (EntityType.BASELINE, extraction.baselines),
        (EntityType.ASSUMPTION, extraction.assumptions),
        (EntityType.FAILURE_MODE, extraction.failure_modes),
    ]


def run(topic: str, max_papers: int | None = None, rebuild: bool = True) -> PipelineResult:
    max_papers = max_papers or config.max_papers
    cache = Cache(config.cache_path)
    graph = GraphManager()
    vectors = VectorManager()
    resolver = Resolver()
    result = PipelineResult(topic=topic)

    graph.setup_constraints()
    if rebuild:
        graph.clear()

    # 1. PLAN + INGEST ------------------------------------------------------
    print(f"\n[1/7] Planning queries for: {topic!r}")
    qset = query_planner.plan_queries(topic, cache)
    print(f"      {len(qset.all_queries())} queries across surveys/seminal/frontier")
    print(f"[2/7] Ingesting papers (max {max_papers}) ...")
    papers = paper_fetcher.ingest(qset.all_queries(), max_papers, cache)
    result.n_papers = len(papers)
    print(f"      {len(papers)} papers after dedupe")
    if not papers:
        print("      No papers fetched — check connectivity or widen the topic.")
        graph.close()
        return result

    # 2. EXTRACT + VERIFY + CANONICALIZE + BUILD ----------------------------
    print(f"[3/7] Extracting with local model '{config.llm_model}' (cached) ...")
    paper_ids = {p.paper_id for p in papers}
    for p in tqdm(papers, desc="      extract"):
        graph.add_paper(p.paper_id, p.title, p.year, p.citation_count)
        vectors.add_paper(p.paper_id, p.title, p.abstract, p.year)
        # citation edges (only to papers in-corpus, to keep the graph dense & relevant)
        for ref in p.references:
            if ref in paper_ids:
                graph.add_citation(p.paper_id, ref)

        ext = extractor.extract_paper(p, cache)
        ext, _ = verifier.verify(ext, p)

        # canonicalize + add typed entities
        canon: dict[str, dict[str, str]] = {t.value: {} for t in EntityType}
        for etype, names in _type_lists(ext):
            for raw in names:
                c = resolver.resolve(raw, etype)
                canon[etype.value][raw] = c
                graph.add_mention(p.paper_id, c, etype.value)

        # typed relationships from reified results (method APPLIED_TO component, etc.)
        for r in ext.results:
            if r.method and r.component:
                graph.add_edge(resolver.resolve(r.method, EntityType.METHOD), "Method",
                               RelType.APPLIED_TO,
                               resolver.resolve(r.component, EntityType.COMPONENT), "Component",
                               p.paper_id, p.year)
            if r.method and r.dataset:
                graph.add_edge(resolver.resolve(r.method, EntityType.METHOD), "Method",
                               RelType.EVALUATED_ON,
                               resolver.resolve(r.dataset, EntityType.DATASET), "Dataset",
                               p.paper_id, p.year)
            if r.method and r.metric:
                graph.add_edge(resolver.resolve(r.method, EntityType.METHOD), "Method",
                               RelType.ACHIEVES,
                               resolver.resolve(r.metric, EntityType.METRIC), "Metric",
                               p.paper_id, p.year)
            graph.add_result(p.paper_id,
                             resolver.resolve(r.method, EntityType.METHOD) if r.method else "",
                             resolver.resolve(r.component, EntityType.COMPONENT) if r.component else "",
                             resolver.resolve(r.dataset, EntityType.DATASET) if r.dataset else "",
                             r.metric, r.value)

        # Co-occurrence fallback edges: method APPLIED_TO each component mentioned together.
        # This seeds the bipartite projections even when the model didn't emit explicit results.
        for m in canon["Method"].values():
            for comp in canon["Component"].values():
                graph.add_edge(m, "Method", RelType.APPLIED_TO, comp, "Component",
                               p.paper_id, p.year)
            for ds in canon["Dataset"].values():
                graph.add_edge(m, "Method", RelType.EVALUATED_ON, ds, "Dataset",
                               p.paper_id, p.year)
        for comp in canon["Component"].values():
            for fm in canon["FailureMode"].values():
                graph.add_edge(comp, "Component", RelType.HAS_FAILURE_MODE, fm, "FailureMode",
                               p.paper_id, p.year)

    result.n_entities = sum(resolver.size().values())
    print(f"      {result.n_entities} canonical entities across types: {resolver.size()}")

    # 3. GENERATE -----------------------------------------------------------
    print("[4/7] Generating structural gap candidates ...")
    all_edges = graph.all_entity_edges()  # fetch once, reuse across networkx generators
    cands = []
    cands += bipartite.generate(graph)
    cands += paths.generate(graph, edges=all_edges)
    bridges, dense = communities.generate(graph, edges=all_edges)
    cands += bridges
    cands += sparsity.generate(graph)
    result.dense_clusters = dense
    print(f"      {len(cands)} raw candidates "
          f"(voids/paths/bridges/sparsity); {len(dense)} dense clusters flagged")

    # 4. FILTER -------------------------------------------------------------
    print("[5/7] Filtering (structural analogy + evidence weighting) ...")
    cands = analogy.filter_candidates(cands, graph)
    cands = evidence.score_candidates(cands, graph)
    # de-dupe by entity set, keep best-scored
    seen, unique = set(), []
    for c in cands:
        if c.key() in seen:
            continue
        seen.add(c.key())
        unique.append(c)
    print(f"      {len(unique)} candidates after filtering")

    # 5. NARRATE + 6. FALSIFY ----------------------------------------------
    print("[6/7] Narrating + falsifying top candidates ...")
    detected = gap_detector.narrate(unique, graph, top_k=15)
    checked = falsifier.falsify(detected, vectors, k=5)
    surv = falsifier.survivors(checked, min_significance=2)
    result.survivors = surv
    print(f"      {len(surv)}/{len(checked)} gaps survived falsification")

    # 7. TOPICS -------------------------------------------------------------
    print("[7/7] Generating grounded research topics ...")
    result.topics = topic_generator.generate_topics(surv, graph, top_k=10)
    print(f"      {len(result.topics)} topics generated\n")

    graph.close()
    cache.close()
    return result
