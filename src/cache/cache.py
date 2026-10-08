"""
Content-addressed cache — the primary cost/compute control and the reproducibility backbone.

Principle: key every expensive result by (immutable inputs + version tags of the code that
produced it). When nothing changed, read from cache; when an input or a version tag changes,
the key changes and only that computation is redone. It's build-system logic (Make/Bazel)
applied to LLM calls and web fetches.

Why it matters here: with a local model the marginal token is free, but each extraction still
costs *seconds of GPU time*. Re-running the pipeline over the same corpus during development is
the real time sink. Pay once per (paper, prompt_version, model, schema_version), then never again.

Storage: one SQLite file. No server, transactional, trivially portable. The graph/vector stores
hold the *working* state and can be wiped and rebuilt from this cache for free; this cache is the
thing you don't delete.
"""
from __future__ import annotations
import sqlite3
import hashlib
import json
import os
import threading
from typing import Any, Optional


class Cache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  k TEXT PRIMARY KEY,"
            "  v TEXT NOT NULL,"
            "  created_at TEXT DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        self._conn.commit()

    @staticmethod
    def key(namespace: str, *parts: Any) -> str:
        """Build a stable key from a namespace and any JSON-serializable parts.

        Pass the version tags as parts, e.g.:
            Cache.key("extract", paper_id, model, prompt_version, schema_version)
        """
        blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:40]
        return f"{namespace}:{digest}"

    def get(self, key: str) -> Optional[Any]:
        cur = self._conn.execute("SELECT v FROM cache WHERE k = ?", (key,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value, default=str, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (key, payload)
            )
            self._conn.commit()

    def get_or_compute(self, key: str, compute_fn):
        """Return cached value for key, or run compute_fn(), store, and return it."""
        hit = self.get(key)
        if hit is not None:
            return hit, True
        value = compute_fn()
        self.set(key, value)
        return value, False

    def stats(self) -> dict:
        cur = self._conn.execute("SELECT count(*) FROM cache")
        return {"entries": cur.fetchone()[0]}

    def close(self) -> None:
        self._conn.close()
