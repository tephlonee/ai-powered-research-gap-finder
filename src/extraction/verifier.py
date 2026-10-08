"""
Extraction verifier — a cheap faithfulness check, NOT an LLM judge.

Because the entity's evidence sentence is supposed to come from the paper, we have checkable
ground truth: does that sentence actually appear in the source text? Entities whose evidence
can't be found are likely hallucinated and are dropped before they reach the graph. This stops
bad edges from poisoning every downstream gap query.

Uses fuzzy matching (rapidfuzz) so minor whitespace/quoting differences don't cause false drops.
"""
from __future__ import annotations

from rapidfuzz import fuzz

from src.ingestion.paper_fetcher import Paper
from schema.extraction_schema import PaperExtraction, Entity


_MATCH_THRESHOLD = 80  # partial-ratio score; evidence must substantially appear in the text


def _supported(evidence: str, haystack: str) -> bool:
    evidence = (evidence or "").strip()
    if not evidence:
        return False
    if evidence.lower() in haystack:
        return True
    # partial_ratio handles the evidence being a substring with small differences
    return fuzz.partial_ratio(evidence.lower(), haystack) >= _MATCH_THRESHOLD


def verify(extraction: PaperExtraction, paper: Paper) -> tuple[PaperExtraction, int]:
    """Return (filtered_extraction, n_dropped). Only entities are gated; list fields are kept."""
    haystack = f"{paper.title}\n{paper.abstract}\n{paper.full_text}".lower()
    kept: list[Entity] = []
    dropped = 0
    for ent in extraction.entities:
        # If the model gave no evidence, keep the entity but don't claim provenance;
        # if it gave evidence, it must check out.
        if not ent.evidence.strip() or _supported(ent.evidence, haystack):
            kept.append(ent)
        else:
            dropped += 1
    extraction.entities = kept
    return extraction, dropped
