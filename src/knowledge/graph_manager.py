"""
Graph manager — Neo4j knowledge graph where gap-finding actually lives.

Key design choices (fixing the old code's problems):
  * FIXED relationship vocabulary (schema.RelType). The LLM only ever supplies node *values*;
    edge *types* come from a closed enum — no Cypher injection, no label explosion.
  * One generic :Entity label with a `type` property, instead of interpolating dynamic labels.
  * Reified :Result nodes bind method+component+dataset+metric into one fact (enables
    contradiction + evaluation-gap detection later).
  * Every entity-bearing edge carries the source paper id and its year, so the graph is
    time-aware (hot vs. maturing) and every edge is traceable.
  * Citation edges (:Paper)-[:CITES]->(:Paper) are written, enabling citation snowballing later.

The generators read the graph back out (as edge lists) and do set/graph logic in networkx, so no
Neo4j GDS plugin is required — keeps the Docker setup to a single service.
"""
from __future__ import annotations
from typing import Optional

from neo4j import GraphDatabase

from config import config
from schema.extraction_schema import RelType


class GraphManager:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None):
        self._driver = GraphDatabase.driver(
            uri or config.neo4j_uri,
            auth=(user or config.neo4j_user, password or config.neo4j_password),
        )

    def close(self):
        self._driver.close()

    def health_check(self) -> tuple[bool, str]:
        try:
            with self._driver.session() as s:
                s.run("RETURN 1").single()
            return True, "Neo4j OK."
        except Exception as exc:  # noqa: BLE001
            return False, (f"Could not reach Neo4j at {config.neo4j_uri}: {exc}\n"
                          "Start it with `docker compose up -d neo4j`.")

    def setup_constraints(self):
        with self._driver.session() as s:
            s.run("CREATE CONSTRAINT paper_id IF NOT EXISTS "
                  "FOR (p:Paper) REQUIRE p.id IS UNIQUE")
            s.run("CREATE CONSTRAINT entity_key IF NOT EXISTS "
                  "FOR (e:Entity) REQUIRE (e.name, e.type) IS NODE KEY")

    def clear(self):
        with self._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    # --- writes --------------------------------------------------------------
    def add_paper(self, paper_id: str, title: str, year: Optional[int], citations: int):
        with self._driver.session() as s:
            s.run(
                "MERGE (p:Paper {id:$id}) "
                "SET p.title=$title, p.year=$year, p.citations=$citations",
                id=paper_id, title=title, year=year, citations=citations,
            )

    def add_entity(self, name: str, etype: str):
        with self._driver.session() as s:
            s.run("MERGE (e:Entity {name:$name, type:$type})", name=name, type=etype)

    def add_mention(self, paper_id: str, name: str, etype: str):
        with self._driver.session() as s:
            s.run(
                "MATCH (p:Paper {id:$pid}) "
                "MERGE (e:Entity {name:$name, type:$type}) "
                "MERGE (p)-[:MENTIONS]->(e)",
                pid=paper_id, name=name, type=etype,
            )

    def add_edge(self, src_name: str, src_type: str, rel: RelType,
                 dst_name: str, dst_type: str, paper_id: str, year: Optional[int]):
        """Typed, provenance-stamped edge between two entities. rel must be a RelType."""
        with self._driver.session() as s:
            s.run(
                f"MERGE (a:Entity {{name:$a, type:$at}}) "
                f"MERGE (b:Entity {{name:$b, type:$bt}}) "
                f"MERGE (a)-[r:{rel.value}]->(b) "
                f"ON CREATE SET r.papers=[$pid], r.years=[$yr] "
                f"ON MATCH SET r.papers = CASE WHEN $pid IN r.papers THEN r.papers "
                f"            ELSE r.papers + $pid END, "
                f"            r.years = CASE WHEN $yr IN r.years THEN r.years "
                f"            ELSE r.years + $yr END",
                a=src_name, at=src_type, b=dst_name, bt=dst_type,
                pid=paper_id, yr=year,
            )

    def add_result(self, paper_id: str, method: str, component: str, dataset: str,
                   metric: str, value: str):
        with self._driver.session() as s:
            s.run(
                "MATCH (p:Paper {id:$pid}) "
                "CREATE (res:Result {method:$m, component:$c, dataset:$d, "
                "                    metric:$mt, value:$v}) "
                "MERGE (p)-[:REPORTS]->(res)",
                pid=paper_id, m=method, c=component, d=dataset, mt=metric, v=value,
            )

    def add_citation(self, src_paper_id: str, dst_paper_id: str):
        with self._driver.session() as s:
            s.run(
                "MATCH (a:Paper {id:$a}) "
                "MERGE (b:Paper {id:$b}) "
                "MERGE (a)-[:CITES]->(b)",
                a=src_paper_id, b=dst_paper_id,
            )

    # --- reads (for generators) ---------------------------------------------
    def entities_by_type(self, etype: str) -> list[str]:
        with self._driver.session() as s:
            res = s.run("MATCH (e:Entity {type:$t}) RETURN e.name AS name", t=etype)
            return [r["name"] for r in res]

    def edges(self, rel: RelType) -> list[tuple[str, str, list[str]]]:
        """Return [(src_name, dst_name, [paper_ids])] for a relationship type."""
        with self._driver.session() as s:
            res = s.run(
                f"MATCH (a:Entity)-[r:{rel.value}]->(b:Entity) "
                f"RETURN a.name AS a, b.name AS b, r.papers AS papers"
            )
            return [(r["a"], r["b"], r["papers"] or []) for r in res]

    def entity_neighbors(self, name: str) -> set[str]:
        """All entities adjacent to `name` across any typed relationship (undirected)."""
        with self._driver.session() as s:
            res = s.run(
                "MATCH (e:Entity {name:$n})--(m:Entity) RETURN DISTINCT m.name AS n2",
                n=name,
            )
            return {r["n2"] for r in res}

    def entity_degrees(self) -> dict[str, int]:
        with self._driver.session() as s:
            res = s.run(
                "MATCH (e:Entity) "
                "OPTIONAL MATCH (e)--(m:Entity) "
                "RETURN e.name AS name, count(DISTINCT m) AS deg"
            )
            return {r["name"]: r["deg"] for r in res}

    def paper_meta(self) -> dict[str, dict]:
        with self._driver.session() as s:
            res = s.run("MATCH (p:Paper) RETURN p.id AS id, p.year AS year, "
                       "p.citations AS citations, p.title AS title")
            return {r["id"]: {"year": r["year"], "citations": r["citations"],
                              "title": r["title"]} for r in res}

    def all_entity_edges(self) -> list[tuple[str, str]]:
        """Every entity-entity edge (any type), undirected pairs — for networkx projection."""
        with self._driver.session() as s:
            res = s.run("MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name AS a, b.name AS b")
            return [(r["a"], r["b"]) for r in res]
