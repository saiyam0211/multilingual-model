"""IndicTrans2-based translation wrapper with LRU caching.

Used to translate Indic prompts → English for the cross-lingual differential
reward signal (Phase A1). The English back-translation is used to probe the
target model's English-side refusal, giving us the core gap_indicator signal.

Design:
  - Uses AI4Bharat/IndicTrans2-distilled-200M for speed (+4-8 BLEU vs NLLB on Indic↔EN).
  - Falls back to deep_translator GoogleTranslator when model unavailable (CPU dev mode).
  - LRU cache keyed on (prompt_hash, lang) → avoids re-translating during GRPO rollouts.
  - Thread-safe: cache has a lock for concurrent env calls.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from functools import lru_cache

from ..config import settings
from ..utils.logging import get_logger

log = get_logger(__name__)

# IndicTrans2 language code mapping to BCP-47
_INDIC_TO_FLORES = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "en": "eng_Latn",
}

# deep_translator lang codes
_LANG_TO_GOOGLE = {
    "hi": "hi",
    "ta": "ta",
    "bn": "bn",
    "mr": "mr",
    "te": "te",
    "kn": "kn",
    "en": "en",
}


class TranslationCache:
    """Bounded LRU cache for translations. Thread-safe."""

    def __init__(self, max_size: int = 2048) -> None:
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max = max_size
        self._hits = 0
        self._misses = 0

    def _key(self, text: str, src_lang: str, tgt_lang: str) -> str:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
        return f"{src_lang}_{tgt_lang}_{h}"

    def get(self, text: str, src_lang: str, tgt_lang: str) -> str | None:
        k = self._key(text, src_lang, tgt_lang)
        with self._lock:
            val = self._cache.get(k)
            if val is not None:
                self._cache.move_to_end(k)
                self._hits += 1
                return val
            self._misses += 1
            return None

    def put(self, text: str, src_lang: str, tgt_lang: str, translation: str) -> None:
        k = self._key(text, src_lang, tgt_lang)
        with self._lock:
            self._cache[k] = translation
            self._cache.move_to_end(k)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)


class Translator:
    """Indic ↔ English translator with caching.

    Uses deep_translator as the default backend (no GPU needed, works everywhere).
    For production/training, IndicTrans2 can be loaded via `use_indictrans=True`.
    """

    def __init__(self, cache_size: int = 2048, use_indictrans: bool = False) -> None:
        self._cache = TranslationCache(max_size=cache_size)
        self._use_indictrans = use_indictrans and not settings.mock_gpu
        self._indictrans_model = None
        self._indictrans_tokenizer = None
        log.info(
            "translator_init",
            backend="indictrans2" if self._use_indictrans else "google_translate",
            cache_size=cache_size,
        )

    @lru_cache(maxsize=1)
    def _load_indictrans(self):
        """Lazy load IndicTrans2 model. Only on GPU."""
        try:
            import os

            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = settings.translator_model_id
            hf_token = os.environ.get("HF_TOKEN")
            log.info("loading_indictrans2", model=model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                token=hf_token,
            )
            model.eval()
            return model, tokenizer
        except Exception as e:
            log.warning("indictrans2_load_failed", err=str(e)[:200])
            self._use_indictrans = False
            return None, None

    def _translate_google(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Fallback via deep_translator (Google Translate)."""
        from deep_translator import GoogleTranslator

        src_code = _LANG_TO_GOOGLE.get(src_lang, src_lang)
        tgt_code = _LANG_TO_GOOGLE.get(tgt_lang, tgt_lang)
        try:
            result = GoogleTranslator(source=src_code, target=tgt_code).translate(text)
            return result or text
        except Exception as e:
            log.warning("google_translate_failed", err=str(e)[:200], src=src_lang, tgt=tgt_lang)
            return text

    def _translate_indictrans(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate using IndicTrans2 model."""
        model, tokenizer = self._load_indictrans()
        if model is None:
            return self._translate_google(text, src_lang, tgt_lang)

        import torch

        src_flores = _INDIC_TO_FLORES.get(src_lang)
        tgt_flores = _INDIC_TO_FLORES.get(tgt_lang)
        if not src_flores or not tgt_flores:
            return self._translate_google(text, src_lang, tgt_lang)

        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=4,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_flores),
                )
            result = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return result.strip()
        except Exception as e:
            log.warning("indictrans2_translate_failed", err=str(e)[:200])
            return self._translate_google(text, src_lang, tgt_lang)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Translate text with caching. Returns cached result if available."""
        if not text or not text.strip():
            return ""
        if src_lang == tgt_lang:
            return text

        # Check cache
        cached = self._cache.get(text, src_lang, tgt_lang)
        if cached is not None:
            return cached

        # Translate
        if self._use_indictrans:
            result = self._translate_indictrans(text, src_lang, tgt_lang)
        else:
            result = self._translate_google(text, src_lang, tgt_lang)

        # Cache
        self._cache.put(text, src_lang, tgt_lang, result)
        return result

    def indic_to_en(self, text: str, src_lang: str) -> str:
        """Convenience: translate Indic → English."""
        return self.translate(text, src_lang=src_lang, tgt_lang="en")

    def en_to_indic(self, text: str, tgt_lang: str) -> str:
        """Convenience: translate English → Indic."""
        return self.translate(text, src_lang="en", tgt_lang=tgt_lang)

    @property
    def cache_hit_rate(self) -> float:
        return self._cache.hit_rate
