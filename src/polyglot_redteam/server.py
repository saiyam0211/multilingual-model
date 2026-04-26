"""FastAPI API + Gradio UI for HF Space."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException

from .config import settings
from .episode import EpisodeStore, sample_episode
from .frontend import CSS, THEME, create_demo
from .reward import (
    ClusterNoveltyScorer,
    CrossLingualReward,
    DiversityTracker,
    NoveltyStore,
    Translator,
    compose_reward,
)
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

# v3 reward components (initialized lazily)
_translator = None
_cross_lingual = None
_diversity_tracker = None
_cluster_novelty = None
_v3_initialized = False


def _ensure_target():
    """Lazily initialize target client.

    NOTE: When mounting sub-apps, lifespan can occasionally not initialize in
    the expected order on Spaces restarts. This guard prevents /step 500s from
    `_target is None` and recovers automatically.
    """
    global _target
    if _target is None:
        _target = get_target_client()
        log.info("target_lazy_initialized", target=settings.target_model_id)
    return _target


def _ensure_v3():
    """Lazily initialize v3 reward components."""
    global _translator, _cross_lingual, _diversity_tracker, _cluster_novelty, _v3_initialized

    if _v3_initialized:
        return

    # Translator
    _translator = Translator(
        cache_size=settings.translator_cache_size,
        use_indictrans=settings.use_indictrans,
    )

    # Cross-lingual reward (needs translator + target client)
    target = _ensure_target()
    _cross_lingual = CrossLingualReward(translator=_translator, target_client=target)

    # Diversity tracker
    _diversity_tracker = DiversityTracker(window_size=settings.diversity_window_size)

    # Cluster novelty scorer — load confirmed gaps if available
    _cluster_novelty = ClusterNoveltyScorer()
    vuln_path = Path(settings.vulnerability_dataset_path)
    if vuln_path.exists():
        try:
            records = []
            with vuln_path.open() as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            _cluster_novelty.load_confirmed_gaps(records)
            log.info("cluster_novelty_gaps_loaded", path=str(vuln_path), count=len(records))
        except Exception as e:
            log.warning("cluster_novelty_load_failed", err=str(e)[:200])
    else:
        log.info("cluster_novelty_no_dataset", path=str(vuln_path))

    _v3_initialized = True
    log.info("v3_reward_stack_initialized")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _target
    set_global_seed(0)
    _target = get_target_client()
    _ensure_v3()
    log.info("env_started", mock_gpu=settings.mock_gpu, target=settings.target_model_id)
    yield


api = FastAPI(title="polyglot-redteam-api", version="0.3.0", lifespan=lifespan)


@api.get("/health", response_model=HealthResult)
async def health() -> HealthResult:
    import time

    try:
        target = _ensure_target()
        spec = sample_episode(seed=0)
        t0 = time.perf_counter()
        response, _ = await target.generate("नमस्ते", spec.system_prompt)
        rt_ms = (time.perf_counter() - t0) * 1000.0
        # Health check uses legacy reward (no cross-lingual probe needed)
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


@api.post("/reset", response_model=EpisodeSpec)
async def reset(req: ResetRequest) -> EpisodeSpec:
    spec = sample_episode(req.seed)
    _episodes.put(spec)
    return spec


@api.post("/step", response_model=StepResult)
async def step(req: StepRequest) -> StepResult:
    spec = _episodes.get(req.episode_id)
    if spec is None:
        raise HTTPException(404, f"unknown episode_id {req.episode_id!r} — call /reset first")

    target = _ensure_target()
    _ensure_v3()

    response, target_latency_ms = await target.generate(req.action, spec.system_prompt)

    # Use v3 multiplicative reward with cross-lingual signal
    breakdown = compose_reward(
        prompt=req.action,
        response=response,
        target_lang=spec.target_lang,
        episode_id=req.episode_id,
        category=spec.category,
        novelty_store=_novelty,
        cross_lingual=_cross_lingual,
        diversity_tracker=_diversity_tracker,
        cluster_novelty=_cluster_novelty,
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


@api.get("/state")
async def state() -> dict:
    return {
        "episodes_in_store": len(_episodes._store),
        "novelty_corpus_size": len(_novelty._store),
        "mock_gpu": settings.mock_gpu,
        "v3_initialized": _v3_initialized,
        "translator_cache_hit_rate": _translator.cache_hit_rate if _translator else 0.0,
    }


# Root app: mount API under /api and Gradio at /
app = FastAPI(title="polyglot-redteam")
app.mount("/api", api)
gradio_app = create_demo()
app = gr.mount_gradio_app(app, gradio_app, path="/", theme=THEME, css=CSS)
