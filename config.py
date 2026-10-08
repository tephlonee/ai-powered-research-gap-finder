"""
Central configuration — the one place model names, endpoints, and version tags live.

Everything is local: LLM via Ollama, embeddings via sentence-transformers,
graph in Neo4j (Docker), vectors in embedded ChromaDB. No API keys, no per-token cost.

Override any value with an environment variable (see names below) or a .env file.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # dotenv is optional
    pass


@dataclass(frozen=True)
class Config:
    # --- LLM: Ollama (local) -------------------------------------------------
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # qwen2.5:14b-instruct at Q4 is ~9 GB and fits a 16 GB GPU with room for an
    # 8k context. Drop to "qwen2.5:7b-instruct" (~4.7 GB) for speed or an 8 GB card.
    llm_model: str = os.getenv("GAP_LLM_MODEL", "qwen2.5:14b-instruct")
    llm_temperature: float = float(os.getenv("GAP_LLM_TEMPERATURE", "0.0"))
    llm_num_ctx: int = int(os.getenv("GAP_LLM_NUM_CTX", "8192"))

    # --- Embeddings: sentence-transformers (local, tiny, CPU-friendly) -------
    # Runs in-process and barely touches VRAM, so it never competes with Ollama.
    embed_model: str = os.getenv("GAP_EMBED_MODEL", "all-MiniLM-L6-v2")

    # --- Stores --------------------------------------------------------------
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "gapfinder")
    chroma_path: str = os.getenv("GAP_CHROMA_PATH", "./data/chroma")

    # --- Content-addressed cache ---------------------------------------------
    cache_path: str = os.getenv("GAP_CACHE_PATH", "./data/cache.sqlite")

    # --- Version tags: bump to deliberately invalidate caches ----------------
    # The extraction cache key includes these. Change the schema or the prompt
    # and only the affected computations are redone — nothing else.
    schema_version: str = "1.0.0"
    extract_prompt_version: str = "1.0.0"

    # --- Ingestion -----------------------------------------------------------
    max_papers: int = int(os.getenv("GAP_MAX_PAPERS", "100"))
    semantic_scholar_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    request_timeout: int = int(os.getenv("GAP_HTTP_TIMEOUT", "30"))


config = Config()
