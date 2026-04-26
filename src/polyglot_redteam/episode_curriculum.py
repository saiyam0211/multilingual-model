"""Episode curriculum — per-cell yield tracker and hard-prompt replay buffer.

Replaces uniform sampling in episode.py with adaptive sampling that focuses
training on the hardest (lang × cat) cells.

Design:
  - Rolling mean R_total per (lang × cat) cell over last N=200 rollouts.
  - Sampling weight ∝ (target_yield - current_yield + ε).
  - Capped at 5x uniform to prevent cell collapse.
  - Hard-prompt replay buffer: ring buffer of near-miss prompts (R_attack ∈ [0.05, 0.4]).
  - Persist state to state/curriculum.json for training resumes.
"""
from __future__ import annotations

import json
import random
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import get_args

from .config import settings
from .schemas import Category, Language
from .utils.logging import get_logger

log = get_logger(__name__)

LANGS: tuple[Language, ...] = tuple(lang for lang in get_args(Language) if lang != "en")
CATEGORIES: tuple[Category, ...] = get_args(Category)


class CellStats:
    """Rolling statistics for a single (lang × cat) cell."""

    def __init__(self, window: int = 200) -> None:
        self._rewards: deque[float] = deque(maxlen=window)

    def record(self, reward: float) -> None:
        self._rewards.append(reward)

    @property
    def mean_yield(self) -> float:
        if not self._rewards:
            return 0.0
        # Yield = fraction of rewards > 0.5 (successful attacks)
        return sum(1 for r in self._rewards if r > 0.5) / len(self._rewards)

    @property
    def mean_reward(self) -> float:
        if not self._rewards:
            return 0.0
        return sum(self._rewards) / len(self._rewards)

    @property
    def count(self) -> int:
        return len(self._rewards)


class HardPromptBuffer:
    """Ring buffer of near-miss prompts for hard mining.

    Stores prompts with R_attack ∈ [0.05, 0.4] — close to succeeding
    but not quite there. Training revisits these with p=0.25.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._lock = threading.Lock()
        self._buffer: deque[dict] = deque(maxlen=max_size)

    def add(self, prompt: str, lang: str, category: str, r_attack: float) -> None:
        """Add a near-miss prompt if it qualifies."""
        if 0.05 <= r_attack <= 0.4 and prompt.strip():
            with self._lock:
                self._buffer.append({
                    "prompt": prompt,
                    "lang": lang,
                    "category": category,
                    "r_attack": r_attack,
                })

    def sample(self, rng: random.Random) -> dict | None:
        """Sample a random near-miss prompt from the buffer."""
        with self._lock:
            if not self._buffer:
                return None
            return rng.choice(list(self._buffer))

    @property
    def size(self) -> int:
        return len(self._buffer)


class EpisodeCurriculum:
    """Adaptive (lang × cat) sampling with yield-based weighting.

    Cells where the current yield is lower than the target get sampled more often.
    This focuses training on the hardest cells rather than wasting compute on
    cells that already succeed.
    """

    def __init__(
        self,
        target_yield: float | None = None,
        max_weight: float | None = None,
        epsilon: float | None = None,
        window: int | None = None,
    ) -> None:
        self._target_yield = target_yield or settings.curriculum_target_yield
        self._max_weight = max_weight or settings.curriculum_max_weight
        self._epsilon = epsilon or settings.curriculum_epsilon
        self._window = window or settings.curriculum_window
        self._lock = threading.Lock()

        # Per-cell stats
        self._cells: dict[tuple[str, str], CellStats] = {}
        for lang in LANGS:
            for cat in CATEGORIES:
                self._cells[(lang, cat)] = CellStats(window=self._window)

        # Hard prompt buffer
        self.hard_buffer = HardPromptBuffer(max_size=256)

        log.info(
            "curriculum_init",
            target_yield=self._target_yield,
            max_weight=self._max_weight,
            cells=len(self._cells),
        )

    def record_outcome(self, lang: str, category: str, reward: float, r_attack: float = 0.0) -> None:
        """Record a training outcome for curriculum updates."""
        key = (lang, category)
        with self._lock:
            if key in self._cells:
                self._cells[key].record(reward)

    def sample_cell(self, rng: random.Random) -> tuple[str, str]:
        """Sample a (lang, cat) cell with yield-weighted probability.

        Cells with lower yield get higher weight (up to max_weight × uniform).
        """
        with self._lock:
            weights = []
            cells = list(self._cells.keys())

            for key in cells:
                current_yield = self._cells[key].mean_yield
                # Higher weight for cells with lower yield
                gap = self._target_yield - current_yield + self._epsilon
                weight = max(gap, self._epsilon)
                # Cap at max_weight × uniform
                uniform = 1.0 / len(cells)
                weight = min(weight, self._max_weight * uniform)
                weights.append(weight)

            # Normalize
            total = sum(weights)
            if total <= 0:
                return rng.choice(cells)
            probs = [w / total for w in weights]

        # Weighted random choice
        return rng.choices(cells, weights=probs, k=1)[0]

    def get_yield_heatmap(self) -> dict[str, dict[str, float]]:
        """Returns per-cell yield as nested dict for logging/plotting."""
        result: dict[str, dict[str, float]] = defaultdict(dict)
        with self._lock:
            for (lang, cat), stats in self._cells.items():
                result[lang][cat] = stats.mean_yield
        return dict(result)

    def get_reward_heatmap(self) -> dict[str, dict[str, float]]:
        """Returns per-cell mean reward as nested dict."""
        result: dict[str, dict[str, float]] = defaultdict(dict)
        with self._lock:
            for (lang, cat), stats in self._cells.items():
                result[lang][cat] = stats.mean_reward
        return dict(result)

    def get_spread(self) -> float:
        """Spread = max yield - min yield across cells. Should shrink over training."""
        with self._lock:
            yields = [stats.mean_yield for stats in self._cells.values() if stats.count > 0]
        if len(yields) < 2:
            return 0.0
        return max(yields) - min(yields)

    def save_state(self, path: str | None = None) -> None:
        """Persist curriculum state to JSON for training resumes."""
        path = path or settings.curriculum_state_path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        state = {}
        with self._lock:
            for (lang, cat), stats in self._cells.items():
                key = f"{lang}_{cat}"
                state[key] = {
                    "mean_yield": stats.mean_yield,
                    "mean_reward": stats.mean_reward,
                    "count": stats.count,
                }
        state["_meta"] = {
            "target_yield": self._target_yield,
            "hard_buffer_size": self.hard_buffer.size,
            "spread": self.get_spread(),
        }

        p.write_text(json.dumps(state, indent=2))
        log.info("curriculum_state_saved", path=str(p))

    def load_state(self, path: str | None = None) -> bool:
        """Load curriculum state from JSON. Returns True if loaded successfully."""
        path = path or settings.curriculum_state_path
        p = Path(path)
        if not p.exists():
            return False

        try:
            state = json.loads(p.read_text())
            # We can't fully restore the deque, but we can log what the previous
            # training achieved per cell for debugging.
            log.info(
                "curriculum_state_loaded",
                path=str(p),
                spread=state.get("_meta", {}).get("spread", "?"),
            )
            return True
        except Exception as e:
            log.warning("curriculum_state_load_failed", err=str(e)[:200])
            return False
