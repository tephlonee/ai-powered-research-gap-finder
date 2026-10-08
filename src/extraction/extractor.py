"""
Extractor — one paper in, a PaperExtraction out, cached per (paper, model, prompt, schema).

The extraction cache is the single biggest time-saver: a paper's content is immutable, so once
extracted it is never re-run unless you deliberately bump a version tag in config. That's what
makes iterating on the rest of the pipeline cheap.

The prompt pulls the synthesis-matrix columns AND asks explicitly for limitations and future-work
sentences — the highest-value gap signal — and for provenance on every entity so the graph stays
traceable.
"""
from __future__ import annotations

from config import config
from src.cache.cache import Cache
from src.ingestion.paper_fetcher import Paper
from src.extraction import llm_client
from schema.extraction_schema import PaperExtraction


_SYSTEM = (
    "You extract structured information from a research paper for a literature-review knowledge "
    "base. Extract ONLY what the text supports; do not invent entities or relationships. For each "
    "field use the surface form as written. Pay special attention to LIMITATIONS and FUTURE WORK "
    "statements and to the scope the authors restrict themselves to — these are the most valuable "
    "signals. For every entity, include the sentence it came from as its 'evidence'. If a field "
    "has nothing, return an empty list."
)


def _prompt(paper: Paper) -> str:
    return (
        "Extract the structured fields defined by the schema from this paper.\n\n"
        f"{paper.text_for_extraction()}\n\n"
        "Remember: tasks, methods, components (the thing a method operates on), datasets, metrics, "
        "baselines, assumptions, failure_modes, limitations, future_work, reified results, and a "
        "flat 'entities' list with type + evidence for each."
    )


def extract_paper(paper: Paper, cache: Cache) -> PaperExtraction:
    key = Cache.key(
        "extract",
        paper.paper_id,
        config.llm_model,
        config.extract_prompt_version,
        config.schema_version,
    )
    cached = cache.get(key)
    if cached is not None:
        return PaperExtraction(**cached)

    try:
        extraction = llm_client.structured(_SYSTEM, _prompt(paper), PaperExtraction)
    except Exception as exc:  # noqa: BLE001
        print(f"  [extract] failed for {paper.paper_id}: {exc}")
        extraction = PaperExtraction()

    cache.set(key, extraction.model_dump())
    return extraction
