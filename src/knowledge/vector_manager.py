"""
Vector manager — embedded ChromaDB + local sentence-transformers embeddings.

Why Chroma instead of the old Weaviate: on a single machine, Chroma is an *embedded*,
persistent store (one directory on disk) — no extra Docker service, no server to babysit.
For corpora of thousands-to-tens-of-thousands of papers that's plenty.

Why sentence-transformers (all-MiniLM-L6-v2): it's tiny, runs on CPU, and barely touches VRAM,
so it never competes with the Ollama model for GPU memory. Embeddings are computed here and
handed to Chroma explicitly (we don't use Chroma's own embedding function), which keeps the
embedding model swappable and the cost at zero.

Used for: paper retrieval (RAG grounding) and the falsifier's "find a paper that already did
this" search.
"""
from __future__ import annotations
from typing import Optional

from config import config


class VectorManager:
    def __init__(self, path: Optional[str] = None, embed_model: Optional[str] = None):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self._client = chromadb.PersistentClient(path=path or config.chroma_path)
        self._col = self._client.get_or_create_collection(
            name="papers", metadata={"hnsw:space": "cosine"}
        )
        self._embedder = SentenceTransformer(embed_model or config.embed_model)

    def embed(self, text: str) -> list[float]:
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    def add_paper(self, paper_id: str, title: str, abstract: str, year: Optional[int]):
        vec = self.embed(f"{title}. {abstract}")
        self._col.upsert(
            ids=[paper_id],
            embeddings=[vec],
            documents=[f"{title}. {abstract}"],
            metadatas=[{"title": title, "year": year or 0}],
        )

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self._col.count() == 0:
            return []
        res = self._col.query(query_embeddings=[self.embed(query)],
                              n_results=min(k, self._col.count()))
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "paper_id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "title": (res["metadatas"][0][i] or {}).get("title", ""),
                "distance": res["distances"][0][i] if res.get("distances") else None,
            })
        return out

    def count(self) -> int:
        return self._col.count()
