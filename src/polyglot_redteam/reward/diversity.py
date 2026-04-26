"""Diversity reward — stylistic diversity and cluster novelty (anti mode-collapse).

Two components per OpenAI Diverse & Effective Red Teaming (2412.18693):

1. **Stylistic diversity (R_diversity)**: measures how different the current prompt
   is from recent attacks in the same (lang, cat) cell. Uses sentence embeddings
   projected away from the goal subspace (§3.4 of the paper). Score is sigmoid-normalized
   1 - max_sim to last K=32 attacks.

2. **Cluster novelty (R_novelty)**: cosine distance from the centroid of confirmed
   gaps for the same (lang, cat) cell. Prevents reward-hacking via replaying known attacks.

Both use the existing paraphrase-multilingual-MiniLM-L12-v2 encoder for speed.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict, defaultdict
from functools import lru_cache

import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _encoder():
    """Shared sentence encoder — loaded once, cached globally."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def _encode(text: str) -> np.ndarray:
    """Encode text to normalized embedding."""
    return _encoder().encode(text, normalize_embeddings=True)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


def _sigmoid(x: float, k: float = 5.0, x0: float = 0.0) -> float:
    """Sigmoid normalization: maps x ∈ R → [0, 1]."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


class DiversityTracker:
    """Tracks recent attacks per (lang, cat) cell for stylistic diversity scoring.

    Maintains a rolling window of K=32 embeddings per cell. New prompts are scored
    against this window: 1 - max_cosine_sim, sigmoid normalized.
    """

    def __init__(self, window_size: int = 32) -> None:
        self._lock = threading.Lock()
        self._window_size = window_size
        # key: (lang, cat) → OrderedDict[prompt_hash, embedding]
        self._cells: dict[tuple[str, str], OrderedDict[str, np.ndarray]] = defaultdict(OrderedDict)

    def score(self, prompt: str, lang: str, category: str) -> float:
        """Stylistic diversity: 1 - max_sim to window, sigmoid-normalized.

        Returns 1.0 if no prior attacks in this cell (maximally novel).
        Returns ≈0 if near-duplicate of a recent attack.
        """
        if not prompt.strip():
            return 0.0

        emb = _encode(prompt)
        cell_key = (lang, category)

        with self._lock:
            cell = self._cells.get(cell_key)
            if not cell:
                return 1.0  # First attack in cell → maximally novel

            sims = [_cosine_sim(emb, v) for v in cell.values()]
            max_sim = max(sims) if sims else 0.0

        # 1 - max_sim: higher = more diverse
        raw_diversity = 1.0 - max_sim
        # Sigmoid normalize: k=5, x0=0 → maps [-1,1] → [0,1] with inflection at 0
        return _sigmoid(raw_diversity, k=5.0, x0=0.3)

    def add(self, prompt: str, lang: str, category: str) -> None:
        """Add a prompt to the rolling window for its cell."""
        if not prompt.strip():
            return

        emb = _encode(prompt)
        cell_key = (lang, category)

        with self._lock:
            cell = self._cells[cell_key]
            h = hash(prompt)
            cell[h] = emb
            cell.move_to_end(h)
            while len(cell) > self._window_size:
                cell.popitem(last=False)


class ClusterNoveltyScorer:
    """Scores novelty against confirmed gap centroids (anti-replay).

    Prevents the policy from replaying known attacks from vulnerability_dataset_final.jsonl.
    Loads confirmed gaps at init, groups by (lang, cat), computes centroids.
    Novel attacks (far from centroid) get higher scores.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key: (lang, cat) → centroid embedding (np.ndarray)
        self._centroids: dict[tuple[str, str], np.ndarray] = {}
        # key: (lang, cat) → list of embeddings (for top-K anchor comparison)
        self._anchors: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        self._max_anchors = 100

    def load_confirmed_gaps(self, records: list[dict]) -> None:
        """Load confirmed gap prompts and compute per-cell centroids.

        Args:
            records: List of dicts with at least 'prompt', 'lang', 'category' keys.
        """
        cell_prompts: dict[tuple[str, str], list[str]] = defaultdict(list)

        for rec in records:
            prompt = rec.get("prompt") or rec.get("indic_prompt", "")
            lang = rec.get("lang") or rec.get("target_lang", "hi")
            cat = rec.get("category", "scam_engineering")
            if prompt.strip():
                cell_prompts[(lang, cat)].append(prompt)

        for cell_key, prompts in cell_prompts.items():
            # Take up to max_anchors prompts per cell
            sample = prompts[: self._max_anchors]
            embeddings = [_encode(p) for p in sample]

            with self._lock:
                self._anchors[cell_key] = embeddings
                if embeddings:
                    stacked = np.stack(embeddings)
                    centroid = stacked.mean(axis=0)
                    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
                    self._centroids[cell_key] = centroid

        log.info(
            "cluster_novelty_loaded",
            cells=len(self._centroids),
            total_anchors=sum(len(v) for v in self._anchors.values()),
        )

    def score(self, prompt: str, lang: str, category: str) -> float:
        """Novelty score: 1 - max_sim_to_closest_anchor, sigmoid-normalized.

        Returns 1.0 if no confirmed gaps for this cell (everything is novel).
        Returns ≈0 if prompt is near-identical to a known gap.
        """
        if not prompt.strip():
            return 0.0

        cell_key = (lang, category)

        with self._lock:
            anchors = self._anchors.get(cell_key)
            if not anchors:
                return 1.0  # No known gaps → everything novel

        emb = _encode(prompt)
        # Compare against individual anchors (not just centroid) for replay detection
        sims = [_cosine_sim(emb, a) for a in anchors]
        max_sim = max(sims)

        # High similarity to known gap → low novelty
        raw_novelty = 1.0 - max_sim
        return _sigmoid(raw_novelty, k=8.0, x0=0.15)
