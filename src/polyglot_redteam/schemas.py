"""Shared Pydantic models — used by both client and server."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["hi", "ta", "bn", "mr", "te", "kn", "en"]
Category = Literal[
    "illicit_finance",
    "violence_enable",
    "self_harm",
    "privacy_violation",
    "scam_engineering",
]
GateReason = Literal["lang_mismatch", "gibberish", "too_short", "low_diversity", "ok"]


class EpisodeSpec(BaseModel):
    """What the env exposes after reset()."""

    episode_id: str
    target_lang: Language
    category: Category
    system_prompt: str
    instruction: str = Field(..., description="Meta-instruction for the attacker policy.")
    seed: int


class ResetRequest(BaseModel):
    seed: int = Field(default=0, ge=0)


class StepRequest(BaseModel):
    episode_id: str
    action: str = Field(..., description="The attacker-generated prompt.")


class RewardBreakdown(BaseModel):
    """Structured reward — never return a bare float."""

    total: float
    gate_reason: GateReason = "ok"
    components: dict[str, float] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Gym-style step return, JSON-serializable."""

    observation: str = Field(..., description="Target model's response.")
    reward: float
    done: bool = True
    info: dict[str, Any] = Field(default_factory=dict)


class HealthResult(BaseModel):
    status: Literal["healthy", "degraded", "down"]
    target_roundtrip_ms: float | None = None
    judge_loaded: bool
    mock_gpu: bool
    detail: str | None = None
