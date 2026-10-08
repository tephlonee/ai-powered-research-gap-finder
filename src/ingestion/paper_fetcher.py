"""
Paper ingestion — arXiv + Semantic Scholar, with the raw responses cached.

Recall is the ceiling on everything downstream (you can only find gaps in the corpus you
ingested), so this stage matters more than its size suggests. Two sources:

  * arXiv           — abstracts + a usable chunk of context, no key needed.
  * Semantic Scholar — abstracts, citation counts, and references (used later for the
                       citation graph and citation snowballing). A free API key raises the
                       rate limit but is optional.

Everything is keyed into the content-addressed cache so re-runs don't re-hit rate-limited APIs
and the corpus is reproducible.
"""
from __future__ import annotations
import time
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

import requests
from pydantic import BaseModel, Field

from config import config
from src.cache.cache import Cache


class Paper(BaseModel):
    paper_id: str                      # canonical id, e.g. "arxiv:2401.01234" or "s2:<hash>"
    title: str
    abstract: str = ""
    year: Optional[int] = None
    authors: list[str] = Field(default_factory=list)
    citation_count: int = 0
    references: list[str] = Field(default_factory=list)  # external ids of cited papers
    url: str = ""
    source: str = ""
    full_text: str = ""                # populated only if a full-text fetch succeeds

    def text_for_extraction(self) -> str:
        """What the extractor reads. Prefers full text; falls back to title+abstract."""
        body = self.full_text.strip() or self.abstract.strip()
        return f"Title: {self.title}\n\nAbstract/Body:\n{body}"


_ARXIV_API = "http://export.arxiv.org/api/query"
_S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"


def _norm_title(t: str) -> str:
    return " ".join(t.lower().split())


def fetch_arxiv(query: str, max_results: int, cache: Cache) -> list[Paper]:
    key = Cache.key("fetch_arxiv", query, max_results)
    cached = cache.get(key)
    if cached is not None:
        return [Paper(**p) for p in cached]

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    url = f"{_ARXIV_API}?{urllib.parse.urlencode(params)}"
    papers: list[Paper] = []
    try:
        resp = requests.get(url, timeout=config.request_timeout)
        resp.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        for entry in root.findall("a:entry", ns):
            arxiv_url = entry.findtext("a:id", default="", namespaces=ns)
            short = arxiv_url.rsplit("/", 1)[-1]
            year = None
            published = entry.findtext("a:published", default="", namespaces=ns)
            if len(published) >= 4 and published[:4].isdigit():
                year = int(published[:4])
            papers.append(
                Paper(
                    paper_id=f"arxiv:{short}",
                    title=" ".join((entry.findtext('a:title', '', ns) or '').split()),
                    abstract=" ".join((entry.findtext('a:summary', '', ns) or '').split()),
                    year=year,
                    authors=[
                        (a.findtext("a:name", "", ns) or "")
                        for a in entry.findall("a:author", ns)
                    ],
                    url=arxiv_url,
                    source="arxiv",
                )
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  [arxiv] fetch failed for '{query}': {exc}")

    cache.set(key, [p.model_dump() for p in papers])
    return papers


def fetch_semantic_scholar(query: str, max_results: int, cache: Cache) -> list[Paper]:
    key = Cache.key("fetch_s2", query, max_results)
    cached = cache.get(key)
    if cached is not None:
        return [Paper(**p) for p in cached]

    fields = "title,abstract,year,citationCount,externalIds,references.externalIds,authors"
    params = {"query": query, "limit": min(max_results, 100), "fields": fields}
    headers = {}
    if config.semantic_scholar_api_key:
        headers["x-api-key"] = config.semantic_scholar_api_key

    papers: list[Paper] = []
    try:
        resp = requests.get(_S2_SEARCH, params=params, headers=headers,
                            timeout=config.request_timeout)
        if resp.status_code == 429:
            time.sleep(3)  # S2 throttles hard without a key; one polite retry
            resp = requests.get(_S2_SEARCH, params=params, headers=headers,
                                timeout=config.request_timeout)
        resp.raise_for_status()
        for item in resp.json().get("data", []) or []:
            ext = item.get("externalIds") or {}
            pid = ext.get("ArXiv") and f"arxiv:{ext['ArXiv']}" or f"s2:{item.get('paperId','')}"
            refs = []
            for r in item.get("references", []) or []:
                rext = (r or {}).get("externalIds") or {}
                if rext.get("ArXiv"):
                    refs.append(f"arxiv:{rext['ArXiv']}")
            papers.append(
                Paper(
                    paper_id=pid,
                    title=" ".join((item.get("title") or "").split()),
                    abstract=" ".join((item.get("abstract") or "").split()),
                    year=item.get("year"),
                    citation_count=item.get("citationCount") or 0,
                    references=refs,
                    authors=[a.get("name", "") for a in (item.get("authors") or [])],
                    source="semantic_scholar",
                )
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  [s2] fetch failed for '{query}': {exc}")

    cache.set(key, [p.model_dump() for p in papers])
    return papers


def dedupe(papers: list[Paper]) -> list[Paper]:
    """Merge by normalized title; prefer the record with more metadata (citations/refs)."""
    best: dict[str, Paper] = {}
    for p in papers:
        k = _norm_title(p.title)
        if not k:
            continue
        cur = best.get(k)
        if cur is None:
            best[k] = p
        else:
            # keep the richer record, union references
            richer = p if (p.citation_count + len(p.references)) > (
                cur.citation_count + len(cur.references)) else cur
            poorer = cur if richer is p else p
            richer.references = sorted(set(richer.references) | set(poorer.references))
            if not richer.abstract:
                richer.abstract = poorer.abstract
            best[k] = richer
    return list(best.values())


def ingest(queries: list[str], max_papers: int, cache: Cache) -> list[Paper]:
    """Run every query against both sources, dedupe, and cap at max_papers."""
    collected: list[Paper] = []
    per_query = max(5, max_papers // max(1, len(queries)))
    for q in queries:
        collected += fetch_arxiv(q, per_query, cache)
        collected += fetch_semantic_scholar(q, per_query, cache)
    merged = dedupe(collected)
    # Papers with no abstract are useless to the extractor; drop them.
    merged = [p for p in merged if p.abstract or p.full_text]
    return merged[:max_papers]
