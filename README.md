# AI Research Gap Finder — Local / Ollama Edition

Finds **research gaps** in a field by building a knowledge graph of its papers and looking for
what's *structurally missing* — then has a local LLM narrate and a falsifier validate the
candidates. Runs entirely on your own machine: **Ollama** for the LLM, **sentence-transformers**
for embeddings, **Neo4j** for the graph, **ChromaDB** (embedded) for vectors. No API keys, no
per-token cost.

This is a first-principles rebuild of the original API-based prototype. The guiding idea:

> **Structure finds the candidates, filters prune them, the LLM only narrates, and the falsifier
> validates.** Nothing is free-associated.

---

## What it does (the pipeline)

```
topic
  │  plan      LLM expands the topic into a structured query set (surveys / seminal / frontier)
  │  ingest    arXiv + Semantic Scholar, deduped, cached
  │  extract   local LLM pulls a domain-general schema from each paper (cached per paper)
  │  verify    drop entities whose evidence sentence isn't in the paper (anti-hallucination)
  │  canon     "RUL" == "remaining useful life"  (entity resolution)
  │  build      Neo4j graph (reified, typed, time-stamped) + Chroma vectors
  ▼
GENERATE candidates ─ bipartite voids · Swanson two-hop · community bridges · sparsity
FILTER            ─ structural analogy · evidence weighting (citations × recency × support)
NARRATE           ─ LLM phrases each structural candidate (with provenance)
FALSIFY           ─ "find the paper that already did this"; survivors are validated gaps
TOPICS            ─ concrete research questions, grounded in real paper IDs
```

Each bipartite projection surfaces a different **gap type**: Method×Component → population/context
(the classic "works on roller bearings, untested on planetary bearings"), Method×Dataset →
benchmark monoculture, Component×FailureMode → coverage.

---

## Hardware

Built and tuned for a **16 GB NVIDIA GPU** (e.g. RTX 5060 Ti 16 GB).

| Component | Where it runs | Resource |
|---|---|---|
| LLM (`qwen2.5:14b-instruct`, Q4 ≈ 9 GB) | GPU via Ollama | the main VRAM user |
| Embeddings (`all-MiniLM-L6-v2`) | CPU, in-process | negligible VRAM |
| Neo4j | Docker container | RAM/CPU |
| ChromaDB | embedded, on disk | RAM/CPU |

The 14B model at Q4 leaves comfortable headroom on 16 GB for an 8k context.
**8 GB card?** Set `GAP_LLM_MODEL=qwen2.5:7b-instruct` in `.env` (≈ 4.7 GB).

> **RTX 50-series (Blackwell):** make sure you're on recent NVIDIA drivers and a current Ollama
> build — Ollama ships its own CUDA runtime, but very new GPUs need an up-to-date driver.

---

## Setup

**1. Install prerequisites**
- [Ollama](https://ollama.com) (running: `ollama serve`)
- Docker + Docker Compose
- Python 3.10+

**2. Pull the model**
```bash
bash scripts/setup_ollama.sh        # or: ollama pull qwen2.5:14b-instruct
```

**3. Start Neo4j**
```bash
docker compose up -d neo4j          # Browser at http://localhost:7474 (neo4j / gapfinder)
```

**4. Python deps + config**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # defaults work out of the box
```

**5. Verify, then run**
```bash
python main.py --check              # checks Ollama + Neo4j + Chroma
python main.py --topic "predictive maintenance in roller bearings" --max-papers 80 --out results.json
```

First run downloads the embedding model (~90 MB) and extracts every paper (a few seconds each on
GPU). **Re-runs are much faster** — extraction is cached per paper.

---

## Verify the reasoning core without the full stack

```bash
pip install pydantic rapidfuzz networkx neo4j
python scripts/selftest.py
```
Builds a tiny synthetic graph and asserts the generators/filters behave (including finding the
planetary-bearing gap). No Ollama/Neo4j/Chroma required.

---

## Configuration

Everything lives in `config.py`, overridable via `.env` / env vars. Key knobs:

| Variable | Default | Purpose |
|---|---|---|
| `GAP_LLM_MODEL` | `qwen2.5:14b-instruct` | the Ollama model; drop to `7b` for less VRAM |
| `GAP_MAX_PAPERS` | `100` | corpus size (recall ceiling) |
| `GAP_LLM_NUM_CTX` | `8192` | context window |
| `SEMANTIC_SCHOLAR_API_KEY` | *(blank)* | optional; raises S2 rate limit |

**Version tags** (`schema_version`, `extract_prompt_version` in `config.py`) are part of the cache
key. Bump one and only the affected work is re-done — a real migration, like a DB migration.

---

## What's implemented vs. deliberately deferred

**Implemented (the runnable core):**
- Versioned, domain-general extraction schema (replaces the old domain-locked OWL ontology)
- Content-addressed SQLite cache (fetch, query-plan, extraction, keyed by version tags)
- Local structured extraction via Ollama constrained decoding + evidence verifier
- Entity canonicalization (acronym map + fuzzy; per type)
- Reified Neo4j graph: fixed relationship vocabulary, timestamps, citation edges
- Generators: **bipartite voids, Swanson two-hop, community bridges, sparsity**
- Filters: **structural analogy, evidence weighting**
- Detector (LLM narrates candidates) + **falsifier** (disconfirm + significance/feasibility)
- Grounded topic generator (references are real paper IDs)

**Deferred (hooks are in place; see the rebuild backlog):**
- Full-text ingestion (currently abstracts; schema + extractor already accept full text via
  `Paper.full_text`)
- Learned link prediction (node2vec/TransE) — the query-based generators cover most value first
- Embedding-based canonicalization (resolver interface is ready for it)
- Retrospective "time-travel" evaluation harness; silver/gold eval sets
- Separate problematization (assumption-inversion) pipeline

---

## Cost

**Local = free.** No API keys, no per-token billing. The only budget is GPU time, and the
content-addressed cache means you pay for each paper's extraction exactly once — re-running the
pipeline over the same corpus during development costs nothing further.

---

## Project layout

```
config.py                     central config + version tags
schema/extraction_schema.py   the domain-general, versioned schema (+ fixed RelType vocab)
src/
  cache/cache.py              content-addressed SQLite cache
  ingestion/                  query_planner.py, paper_fetcher.py
  extraction/                 llm_client.py (Ollama), extractor.py, verifier.py
  canonicalization/resolver.py
  knowledge/                  graph_manager.py (Neo4j), vector_manager.py (Chroma)
  generators/                 bipartite.py, paths.py, communities.py, sparsity.py
  filters/                    evidence.py, analogy.py
  analysis/                   gap_detector.py, falsifier.py, topic_generator.py
  pipeline.py                 wires it all together
main.py                       CLI
scripts/                      setup_ollama.sh, selftest.py
docker-compose.yml            Neo4j only
```
