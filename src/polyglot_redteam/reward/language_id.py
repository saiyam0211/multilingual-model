"""Language identification — fasttext lid.176 (script-aware, content-based for short text)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ..config import settings


@lru_cache(maxsize=1)
def _load_model():
    import fasttext

    path = settings.fasttext_lid_path
    if not Path(path).exists():
        raise FileNotFoundError(
            f"fasttext lid model not found at {path}. "
            f"Run scripts/download_assets.sh"
        )
    fasttext.FastText.eprint = lambda *a, **kw: None
    return fasttext.load_model(path)


def detect_language(text: str) -> tuple[str, float]:
    """Returns (lang_code, confidence). 'und' if input is empty/unparseable."""
    cleaned = text.replace("\n", " ").strip()
    if not cleaned:
        return ("und", 0.0)
    if os.environ.get("LID_OFFLINE_FALLBACK") == "1":
        return _script_fallback(cleaned), 0.95
    try:
        model = _load_model()
    except FileNotFoundError:
        return _script_fallback(cleaned), 0.5
    # NOTE: fasttext-wheel's model.predict() calls np.array(..., copy=False)
    # which numpy>=2 rejects. Bypass by calling the underlying C++ binding,
    # which returns List[Tuple[float, str]] sorted by score desc.
    pairs = model.f.predict(cleaned, 1, 0.0, "")
    if not pairs:
        return _script_fallback(cleaned), 0.5
    prob, label = pairs[0]
    code = label.replace("__label__", "")
    return (code, float(prob))


def is_valid_language(text: str, expected: str, threshold: float | None = None) -> bool:
    """True iff detected lang == expected with confidence ≥ threshold."""
    threshold = threshold if threshold is not None else settings.lang_id_threshold
    code, conf = detect_language(text)
    return code == expected and conf >= threshold


_SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)],   # Devanagari (also covers Marathi by script)
    "mr": [(0x0900, 0x097F)],
    "ta": [(0x0B80, 0x0BFF)],
    "bn": [(0x0980, 0x09FF)],
    "te": [(0x0C00, 0x0C7F)],
    "kn": [(0x0C80, 0x0CFF)],
    "en": [(0x0020, 0x007F)],
}


def _script_fallback(text: str) -> str:
    """Used when fasttext is unavailable. Crude: dominant script wins."""
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        for lang, ranges in _SCRIPT_RANGES.items():
            for lo, hi in ranges:
                if lo <= cp <= hi:
                    counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "und"
    return max(counts.items(), key=lambda kv: kv[1])[0]
