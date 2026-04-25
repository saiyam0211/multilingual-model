"""Novelty bonus — episode-scoped cosine similarity over recent successful prompts."""
from __future__ import annotations

import threading
from collections import OrderedDict
from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


class NoveltyStore:
    """Bounded LRU of (prompt, embedding) tuples. Thread-safe."""

    def __init__(self, max_size: int = 100) -> None:
        self._lock = threading.Lock()
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        self._max = max_size

    def score(self, prompt: str) -> float:
        """Returns 1 - max cosine similarity with stored prompts. 1.0 if empty."""
        if not prompt.strip():
            return 0.0
        emb = _encoder().encode(prompt, normalize_embeddings=True).tolist()
        with self._lock:
            if not self._store:
                return 1.0
            sims = [_dot(emb, v) for v in self._store.values()]
            return max(0.0, 1.0 - max(sims))

    def add(self, prompt: str) -> None:
        emb = _encoder().encode(prompt, normalize_embeddings=True).tolist()
        with self._lock:
            self._store[prompt] = emb
            self._store.move_to_end(prompt)
            while len(self._store) > self._max:
                self._store.popitem(last=False)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
