"""Frozen-target client. Calls HF Inference Provider (default) or custom vLLM endpoint."""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import settings
from ..utils.logging import get_logger

log = get_logger(__name__)


class TargetClient(Protocol):
    async def generate(self, prompt: str, system: str) -> tuple[str, float]:
        """Returns (response_text, latency_ms). Never raises — returns ('', latency) on failure."""
        ...


class HFInferenceTarget:
    """HF Inference Providers client (router). Free for many community models.

    Tries Providers router first (`router.huggingface.co/<provider>/...`); falls
    back to legacy serverless on 404. The router supports more models including
    Qwen, DeepSeek, Mistral via partner inference backends.
    """

    def __init__(self, model_id: str, token: str | None, provider: str = "hf-inference") -> None:
        self.primary_url = (
            f"https://router.huggingface.co/{provider}/models/{model_id}/v1/chat/completions"
        )
        self.fallback_url = (
            f"https://api-inference.huggingface.co/models/{model_id}/v1/chat/completions"
        )
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.model_id = model_id

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        wait=wait_exponential(multiplier=0.5, max=8),
        stop=stop_after_attempt(settings.target_max_retries),
        reraise=False,
    )
    async def _post(self, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=settings.target_timeout_s) as client:
            r = await client.post(self.primary_url, headers=self.headers, json=payload)
            if r.status_code == 404:
                r = await client.post(self.fallback_url, headers=self.headers, json=payload)
            r.raise_for_status()
            return r.json()

    async def generate(self, prompt: str, system: str) -> tuple[str, float]:
        import time

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": settings.target_max_tokens,
            "temperature": settings.target_temperature,
        }
        t0 = time.perf_counter()
        try:
            data = await self._post(payload)
            text = data["choices"][0]["message"]["content"]
        except Exception as e:
            log.warning("target_call_failed", err=str(e), model=self.model_id)
            text = ""
        latency = (time.perf_counter() - t0) * 1000.0
        return text, latency


class MockTarget:
    """Local dev target. Echoes a refusal-ish or compliance-ish response."""

    async def generate(self, prompt: str, system: str) -> tuple[str, float]:
        import asyncio
        import random

        await asyncio.sleep(0.01)
        if random.random() < 0.7:
            return ("I cannot help with that request.", 12.0)
        return (f"Sure, here's what you can try: step 1, then step 2. (mock) prompt={prompt[:30]}", 14.0)


@lru_cache(maxsize=1)
def get_target_client() -> TargetClient:
    if settings.mock_gpu and not settings.hf_token:
        log.info("using_mock_target", reason="MOCK_GPU and no HF_TOKEN")
        return MockTarget()
    if settings.target_provider == "hf-inference":
        return HFInferenceTarget(
            settings.target_model_id, settings.hf_token, provider="hf-inference"
        )
    raise NotImplementedError(f"target provider {settings.target_provider!r} not wired")
