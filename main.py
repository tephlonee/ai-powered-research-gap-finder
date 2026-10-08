"""
CLI entrypoint for the AI Research Gap Finder (local / Ollama edition).

Examples
--------
  # verify Ollama + Neo4j are reachable and the model is pulled
  python main.py --check

  # find gaps for a topic
  python main.py --topic "predictive maintenance in roller bearings" --max-papers 80

  # write full results to JSON
  python main.py --topic "graph neural networks for fault diagnosis" --out results.json
"""
from __future__ import annotations
import argparse
import json
import sys

from config import config


def cmd_check() -> int:
    from src.extraction import llm_client
    from src.knowledge.graph_manager import GraphManager

    ok_llm, msg_llm = llm_client.health_check()
    print(("[ok] " if ok_llm else "[!!] ") + msg_llm)

    graph = GraphManager()
    ok_graph, msg_graph = graph.health_check()
    graph.close()
    print(("[ok] " if ok_graph else "[!!] ") + msg_graph)

    try:
        from src.knowledge.vector_manager import VectorManager
        vm = VectorManager()
        print(f"[ok] Chroma OK ({vm.count()} papers indexed) + embedder loaded.")
        ok_vec = True
    except Exception as exc:  # noqa: BLE001
        print(f"[!!] Vector store/embedder problem: {exc}")
        ok_vec = False

    return 0 if (ok_llm and ok_graph and ok_vec) else 1


def _serialize(result) -> dict:
    return {
        "topic": result.topic,
        "papers": result.n_papers,
        "entities": result.n_entities,
        "dense_clusters": result.dense_clusters,
        "gaps": [
            {
                "title": t.topic.title,
                "question": t.topic.question,
                "approach": t.topic.approach,
                "gap_type": t.topic.gap_type,
                "significance": t.significance,
                "feasibility": t.feasibility,
                "references": t.references,
            }
            for t in result.topics
        ],
    }


def cmd_run(topic: str, max_papers: int, out: str | None, keep: bool) -> int:
    from src.pipeline import run

    result = run(topic, max_papers=max_papers, rebuild=not keep)

    print("=" * 72)
    print(f"RESEARCH GAPS & TOPICS for: {topic}")
    print("=" * 72)
    if not result.topics:
        print("No validated gaps produced. Try more papers (--max-papers) or a broader topic.")
    for i, t in enumerate(result.topics, 1):
        print(f"\n{i}. {t.topic.title}   "
              f"[{t.topic.gap_type} | sig {t.significance}/5, feas {t.feasibility}/5]")
        print(f"   Q: {t.topic.question}")
        print(f"   Approach: {t.topic.approach}")
        if t.references:
            print(f"   Grounded in: {', '.join(t.references[:5])}")

    if result.dense_clusters:
        print("\n(Context) Crowded subtopics to avoid:")
        for d in result.dense_clusters[:3]:
            print(f"   - density {d['density']} ({d['size']} concepts): "
                  f"{', '.join(d['sample'][:5])}")

    if out:
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(_serialize(result), fh, indent=2, ensure_ascii=False)
        print(f"\nFull results written to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="AI Research Gap Finder (local / Ollama)")
    ap.add_argument("--topic", "-t", help="Research topic or area to find gaps in.")
    ap.add_argument("--max-papers", "-n", type=int, default=config.max_papers,
                    help=f"Max papers to ingest (default {config.max_papers}).")
    ap.add_argument("--out", "-o", help="Write full results to this JSON file.")
    ap.add_argument("--keep", action="store_true",
                    help="Keep (don't wipe) the existing graph before running.")
    ap.add_argument("--check", action="store_true",
                    help="Check Ollama/Neo4j/Chroma connectivity and exit.")
    args = ap.parse_args()

    if args.check:
        return cmd_check()
    if not args.topic:
        ap.print_help()
        return 1
    return cmd_run(args.topic, args.max_papers, args.out, args.keep)


if __name__ == "__main__":
    sys.exit(main())
