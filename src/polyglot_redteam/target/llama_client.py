"""Frozen-target client. Uses HF AsyncInferenceClient with provider='auto'."""
from __future__ import annotations

import time
from functools import lru_cache
from typing import Protocol

from huggingface_hub import AsyncInferenceClient
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import settings
from ..utils.logging import get_logger

log = get_logger(__name__)


class TargetClient(Protocol):
    async def generate(self, prompt: str, system: str) -> tuple[str, float]:
        """Returns (response_text, latency_ms). Never raises."""
        ...


class HFInferenceTarget:
    """HF Inference Providers via official SDK. provider='auto' routes to whichever
    partner backend (Together, Fireworks, hf-inference, etc.) serves the model.
    """

    def __init__(self, model_id: str, token: str | None) -> None:
        self.model_id = model_id
        self.client = AsyncInferenceClient(provider="auto", api_key=token)

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=0.5, max=8),
        stop=stop_after_attempt(settings.target_max_retries),
        reraise=True,
    )
    async def _chat(self, messages: list[dict]) -> str:
        out = await self.client.chat_completion(
            model=self.model_id,
            messages=messages,
            max_tokens=settings.target_max_tokens,
            temperature=settings.target_temperature,
        )
        return out.choices[0].message.content or ""

    async def generate(self, prompt: str, system: str) -> tuple[str, float]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        t0 = time.perf_counter()
        try:
            text = await self._chat(messages)
        except Exception as e:
            log.warning("target_call_failed", err=str(e)[:200], model=self.model_id)
            text = ""
        latency = (time.perf_counter() - t0) * 1000.0
        return text, latency


class MockTarget:
    """Local dev fallback when no HF_TOKEN is set."""

    async def generate(self, prompt: str, system: str) -> tuple[str, float]:
        import asyncio
        import random

        await asyncio.sleep(0.01)
        if random.random() < 0.7:
            return ("I cannot help with that request.", 12.0)
        return (f"Sure, here's how. Step 1, then step 2. (mock) prompt={prompt[:30]}", 14.0)


@lru_cache(maxsize=1)
def get_target_client() -> TargetClient:
    if not settings.hf_token or settings.hf_token in {"hf_xxx", ""}:
        log.warning("using_mock_target", reason="no_HF_TOKEN_set")
        return MockTarget()
    if settings.target_provider == "hf-inference":
        log.info("using_hf_target", model=settings.target_model_id)
        return HFInferenceTarget(settings.target_model_id, settings.hf_token)
    raise NotImplementedError(f"target provider {settings.target_provider!r} not wired")
