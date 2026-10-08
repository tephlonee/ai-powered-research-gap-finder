"""
Query planner — turns the user's topic into a *structured query set*, not one naive string.

This is where recall is won or lost. A single query string inherits whatever granularity the
user happened to type and misses vocabulary variance. Instead we ask the local model to produce
three deliberately different sweeps that mirror the human strategic-reading order:

  * surveys  — review/survey queries; surveys pre-aggregate open problems.
  * seminal  — broad queries aimed at foundational, highly-cited work.
  * frontier — specific queries with synonyms and sub-areas for the recent edge.

The LLM only *plans* the queries; deterministic code (paper_fetcher) executes them. That keeps
ingestion debuggable and bounded. The plan is cached per topic, so it's computed once and the
corpus is reproducible.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from config import config
from src.cache.cache import Cache
from src.extraction import llm_client


class QuerySet(BaseModel):
    surveys: list[str] = Field(default_factory=list)
    seminal: list[str] = Field(default_factory=list)
    frontier: list[str] = Field(default_factory=list)

    def all_queries(self) -> list[str]:
        seen, out = set(), []
        for q in self.surveys + self.seminal + self.frontier:
            q = q.strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                out.append(q)
        return out


_SYSTEM = (
    "You are a research librarian building a literature search plan. Given a topic, produce "
    "short keyword queries (2-6 words each) for academic search engines (arXiv, Semantic "
    "Scholar). Produce three groups: 'surveys' (find review/survey papers), 'seminal' (broad "
    "queries that surface foundational, highly-cited work), and 'frontier' (specific queries "
    "with synonyms and sub-areas for the latest work). Use real domain synonyms. Do not invent "
    "unrelated terms. 2-4 queries per group."
)


def plan_queries(topic: str, cache: Cache) -> QuerySet:
    key = Cache.key("query_plan", topic, config.llm_model)
    cached = cache.get(key)
    if cached is not None:
        return QuerySet(**cached)

    prompt = f"Topic: {topic}\n\nProduce the three query groups."
    try:
        qs = llm_client.structured(_SYSTEM, prompt, QuerySet)
    except Exception as exc:  # noqa: BLE001 — never let planning sink the run
        print(f"  [planner] LLM planning failed ({exc}); falling back to the raw topic.")
        qs = QuerySet(surveys=[f"{topic} survey"], seminal=[topic],
                      frontier=[topic])
    # Always include the raw topic as a safety net for recall.
    if topic.lower() not in [q.lower() for q in qs.all_queries()]:
        qs.frontier.append(topic)

    cache.set(key, qs.model_dump())
    return qs
