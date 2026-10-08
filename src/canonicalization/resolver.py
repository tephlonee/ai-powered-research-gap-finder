"""
Entity canonicalization — collapse surface-form variants to one node per real concept.

"RUL", "remaining useful life", and "residual useful life" must become ONE node, or the graph
fragments and every bipartite cell looks falsely empty — manufacturing false-positive gaps.
The old code did MERGE on the raw string, so it had exactly this problem.

v1 strategy (dependency-light, deterministic, good enough to start):
  1. normalize   — lowercase, strip punctuation/whitespace, expand a small acronym map.
  2. fuzzy match — within the same entity TYPE, match to an existing canonical name if the
                   token-set similarity clears a threshold.
Canonicalization is scoped per type (a Method named "CNN" never merges with a Dataset).

Upgrade path (noted, not built): back this with embedding cosine for semantic synonyms that
don't share tokens. The resolver interface stays the same.
"""
from __future__ import annotations
import re

from rapidfuzz import fuzz

from schema.extraction_schema import EntityType


# Seed acronym map; extend freely. Keys/values are compared after normalization.
_ACRONYMS = {
    "rul": "remaining useful life",
    "pdm": "predictive maintenance",
    "cnn": "convolutional neural network",
    "rnn": "recurrent neural network",
    "lstm": "long short-term memory",
    "gnn": "graph neural network",
    "rf": "random forest",
    "svm": "support vector machine",
    "ml": "machine learning",
    "dl": "deep learning",
}

_SIM_THRESHOLD = 90  # token_set_ratio; high, to avoid over-merging distinct concepts


def _normalize(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return _ACRONYMS.get(s, s)


class Resolver:
    """Maintains canonical names per entity type across the whole corpus."""

    def __init__(self):
        # type -> list of (normalized, canonical_display)
        self._canon: dict[str, list[tuple[str, str]]] = {}

    def resolve(self, name: str, etype: EntityType | str) -> str:
        etype = etype.value if isinstance(etype, EntityType) else str(etype)
        norm = _normalize(name)
        if not norm:
            return name.strip()

        bucket = self._canon.setdefault(etype, [])
        # exact normalized hit
        for n, disp in bucket:
            if n == norm:
                return disp
        # fuzzy hit within type
        best, best_disp = 0, None
        for n, disp in bucket:
            score = fuzz.token_set_ratio(norm, n)
            if score > best:
                best, best_disp = score, disp
        if best >= _SIM_THRESHOLD and best_disp is not None:
            return best_disp

        # new canonical concept; display the normalized form (stable, de-noised)
        display = norm
        bucket.append((norm, display))
        return display

    def size(self) -> dict[str, int]:
        return {t: len(v) for t, v in self._canon.items()}
