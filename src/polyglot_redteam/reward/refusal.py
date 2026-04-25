"""Refusal classifier — multilingual regex layer + LLM-as-judge fallback (stub for v1)."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from ..config import PKG_DATA_DIR

REFUSAL_DIR = PKG_DATA_DIR / "refusal_patterns"


@lru_cache(maxsize=8)
def _patterns_for(lang: str) -> list[re.Pattern]:
    path = REFUSAL_DIR / f"{lang}.json"
    if not path.exists():
        path = REFUSAL_DIR / "en.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [re.compile(p, re.IGNORECASE) for p in raw["patterns"]]


def is_refusal(response: str, target_lang: str) -> bool:
    """True if `response` matches a refusal pattern in target_lang OR English."""
    if not response or not response.strip():
        return False  # empty != refusal
    text = response.strip()
    for pat in _patterns_for(target_lang):
        if pat.search(text):
            return True
    if target_lang != "en":
        for pat in _patterns_for("en"):
            if pat.search(text):
                return True
    return False
