"""
Pure-logic self-test — exercises the parts that don't need Ollama/Neo4j/Chroma.

Run:  python scripts/selftest.py
It builds a tiny hand-made "graph" and asserts that the structural generators and filters behave,
so you can validate the reasoning core on any machine before installing the heavy stack.
"""
from __future__ import annotations
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema.extraction_schema import RelType, EntityType  # noqa: E402
from src.cache.cache import Cache  # noqa: E402
from src.canonicalization.resolver import Resolver  # noqa: E402
from src.generators import paths, communities  # noqa: E402
from src.generators.base import GapCandidate  # noqa: E402


class FakeGraph:
    """Minimal stand-in for GraphManager with just the read methods generators/filters use."""
    def __init__(self):
        # Method 'cnn' applied to ball/roller/tapered bearings; NOT planetary (the gap).
        self._applied = [
            ("cnn", "ball bearing", ["p1"]),
            ("cnn", "roller bearing", ["p2"]),
            ("cnn", "tapered bearing", ["p3"]),
            ("svm", "planetary bearing", ["p4"]),
        ]
        self._neighbors = {
            "cnn": {"ball bearing", "roller bearing", "tapered bearing", "cwru"},
            "ball bearing": {"cnn", "cwru"},
            "roller bearing": {"cnn", "cwru"},
            "tapered bearing": {"cnn", "cwru"},
            "planetary bearing": {"svm", "cwru"},
            "svm": {"planetary bearing", "cwru"},
            "cwru": {"ball bearing", "roller bearing", "tapered bearing",
                     "planetary bearing", "cnn", "svm"},
        }
        self._meta = {f"p{i}": {"year": 2020 + i, "citations": 10 * i, "title": f"Paper {i}"}
                      for i in range(1, 6)}

    def edges(self, rel: RelType):
        return self._applied if rel == RelType.APPLIED_TO else []

    def entity_neighbors(self, name):
        return self._neighbors.get(name, set())

    def entity_degrees(self):
        return {k: len(v) for k, v in self._neighbors.items()}

    def paper_meta(self):
        return self._meta

    def all_entity_edges(self):
        out = []
        for a, nbrs in self._neighbors.items():
            for b in nbrs:
                out.append((a, b))
        return out


def check(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def main():
    print("1. Cache (content-addressed, SQLite)")
    with tempfile.TemporaryDirectory() as d:
        c = Cache(os.path.join(d, "c.sqlite"))
        k = Cache.key("ns", "a", 1, "v1")
        c.set(k, {"x": 1})
        check("key is stable", Cache.key("ns", "a", 1, "v1") == k)
        check("round-trips value", c.get(k) == {"x": 1})
        check("version tag changes key", Cache.key("ns", "a", 1, "v2") != k)

    print("2. Canonicalization resolver")
    r = Resolver()
    a = r.resolve("RUL", EntityType.METRIC)
    b = r.resolve("remaining useful life", EntityType.METRIC)
    check("RUL == remaining useful life", a == b)
    check("different concept stays separate",
          r.resolve("f1 score", EntityType.METRIC) != a)

    g = FakeGraph()

    print("3. Bipartite void generator (expects the planetary-bearing gap)")
    from src.generators import bipartite
    voids = bipartite.generate(g)
    pairs = {tuple(v.entities) for v in voids}
    check("finds cnn x planetary bearing void", ("cnn", "planetary bearing") in pairs)
    check("does NOT flag an already-present edge",
          ("cnn", "roller bearing") not in pairs)

    print("4. Structural-analogy filter (planetary is analogous -> kept)")
    from src.filters import analogy
    kept = analogy.filter_candidates(list(voids), g)
    kept_pairs = {tuple(k.entities) for k in kept if k.kind == "void"}
    check("keeps the analogous planetary-bearing gap",
          ("cnn", "planetary bearing") in kept_pairs)

    print("5. Evidence weighting (scores + sorts)")
    from src.filters import evidence
    scored = evidence.score_candidates(list(voids), g)
    check("all candidates scored", all(hasattr(s, "score") for s in scored))
    check("sorted descending", all(scored[i].score >= scored[i + 1].score
                                   for i in range(len(scored) - 1)))

    print("6. Swanson two-hop (shared-neighbour hypotheses)")
    sw = paths.generate(g, min_shared=1, edges=g.all_entity_edges())
    check("produces at least one ABC candidate", len(sw) >= 1)

    print("7. Community / bridge generator runs")
    bridges, dense = communities.generate(g, edges=g.all_entity_edges())
    check("returns (bridges, dense) without error", isinstance(dense, list))

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    main()
