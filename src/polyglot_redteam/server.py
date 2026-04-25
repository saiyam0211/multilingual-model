"""FastAPI app exposing OpenEnv's reset/step/state contract + Gradio UI."""
from __future__ import annotations

from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .config import settings
from .episode import EpisodeStore, sample_episode
from .frontend import CSS, THEME, create_demo
from .reward import NoveltyStore, compose_reward
from .schemas import (
    EpisodeSpec,
    HealthResult,
    ResetRequest,
    StepRequest,
    StepResult,
)
from .target import get_target_client
from .utils.logging import configure_logging, get_logger
from .utils.seed import set_global_seed

configure_logging(json_logs=False)
log = get_logger(__name__)

_episodes = EpisodeStore()
_novelty = NoveltyStore(max_size=100)
_target = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _target
    set_global_seed(0)
    _target = get_target_client()
    log.info("env_started", mock_gpu=settings.mock_gpu, target=settings.target_model_id)
    yield


app = FastAPI(title="polyglot-redteam", version="0.1.0", lifespan=lifespan)


@app.get("/")
async def root():
    return RedirectResponse(url="/ui")


@app.get("/health", response_model=HealthResult)
async def health() -> HealthResult:
    import time

    try:
        spec = sample_episode(seed=0)
        t0 = time.perf_counter()
        response, _ = await _target.generate("नमस्ते", spec.system_prompt)
        rt_ms = (time.perf_counter() - t0) * 1000.0
        _ = compose_reward(
            prompt="नमस्ते दुनिया, यह एक स्वास्थ्य जांच है।",
            response=response or "I cannot help.",
            target_lang="hi",
            episode_id="health",
            category="scam_engineering",
            novelty_store=_novelty,
        )
        return HealthResult(
            status="healthy",
            target_roundtrip_ms=rt_ms,
            judge_loaded=not settings.mock_gpu,
            mock_gpu=settings.mock_gpu,
        )
    except Exception as e:  # noqa: BLE001
        log.error("health_check_failed", err=str(e))
        return HealthResult(
            status="degraded",
            target_roundtrip_ms=None,
            judge_loaded=False,
            mock_gpu=settings.mock_gpu,
            detail=str(e),
        )


@app.post("/reset", response_model=EpisodeSpec)
async def reset(req: ResetRequest) -> EpisodeSpec:
    spec = sample_episode(req.seed)
    _episodes.put(spec)
    return spec


@app.post("/step", response_model=StepResult)
async def step(req: StepRequest) -> StepResult:
    spec = _episodes.get(req.episode_id)
    if spec is None:
        raise HTTPException(404, f"unknown episode_id {req.episode_id!r} — call /reset first")

    response, target_latency_ms = await _target.generate(req.action, spec.system_prompt)
    breakdown = compose_reward(
        prompt=req.action,
        response=response,
        target_lang=spec.target_lang,
        episode_id=req.episode_id,
        category=spec.category,
        novelty_store=_novelty,
    )
    return StepResult(
        observation=response,
        reward=breakdown.total,
        done=True,
        info={
            "gate_reason": breakdown.gate_reason,
            "reward_components": breakdown.components,
            "target_lang": spec.target_lang,
            "category": spec.category,
            "target_latency_ms": target_latency_ms,
        },
    )


@app.get("/state")
async def state() -> dict:
    return {
        "episodes_in_store": len(_episodes._store),
        "novelty_corpus_size": len(_novelty._store),
        "mock_gpu": settings.mock_gpu,
    }


# Mount Gradio AFTER all API routes so they take priority
gradio_app = create_demo()
app = gr.mount_gradio_app(app, gradio_app, path="/ui", theme=THEME, css=CSS)
