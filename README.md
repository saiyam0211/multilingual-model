---
title: Polyglot Red-Teamer
emoji: "🛡"
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: OpenEnv multilingual safety auditing for Indic LLMs
tags:
  - openenv
  - safety
  - red-teaming
  - multilingual
  - rlvr
  - grpo
---

# Polyglot Red-Teamer

> An OpenEnv environment for systematic safety auditing of LLMs in Indian
> languages (Hindi, Tamil, Bengali, Marathi, Telugu, Kannada). Trains a
> Qwen2.5-3B attacker via GRPO to discover prompt patterns that bypass safety
> on a frozen Llama-3.1-8B-Instruct target.

> ⚠️ **Work in progress.** Hackathon submission — Apr 25–26, 2026.

## The problem
Most LLM safety training is in English. Llama-3.1 refuses to give tax-fraud
instructions in English; in Tamil, it sometimes answers. India has 700M+
internet users who prefer their native language over English. We built the
first automated tool to find these gaps.

## What's here
- `src/polyglot_redteam/` — OpenEnv FastAPI server, reward composer, target client
- `notebooks/` — calibration, baseline eval, GRPO training
- `configs/base.yaml` — single source of truth for all hyperparameters
- `tests/` — adversarial probes against the reward function (run before training)
- `docs/ETHICS.md` — what we publish, what we don't, why

## Running locally (Mac / CPU dev)
```bash
source .venv/bin/activate
bash scripts/download_assets.sh   # fasttext lid model (~124MB)
pytest                            # adversarial probes pass
bash scripts/smoke_test.sh        # full env round-trip via MockTarget
```

## Running with real target
Set `HF_TOKEN` in `.env` and either keep `MOCK_GPU=1` (Llama-Guard mocked, real
target via HF Inference) or set `MOCK_GPU=0` on a GPU box for full reward.

## Documents
- [`PLAN.md`](PLAN.md) — strategy
- [`rules.md`](rules.md) — failure-mode bible
- [`SKILL.md`](SKILL.md) — Claude Code agent directives
- [`EXECUTION.md`](EXECUTION.md) — hour-by-hour build plan
- [`DECISIONS.md`](DECISIONS.md) — pre-decided pivot answers
- [`docs/ETHICS.md`](docs/ETHICS.md) — what we publish and what we don't

## Live env
- Space: https://huggingface.co/spaces/Saiyam0211/polyglot-redteam
- Endpoint: `https://saiyam0211-polyglot-redteam.hf.space`
- Health: `curl https://saiyam0211-polyglot-redteam.hf.space/health`

## Status
- [x] Repo scaffold + reward components (B2-B4)
- [x] Adversarial probes pass — 30/30 (DP1)
- [x] HF Space deployment (B5)
- [ ] Baseline eval against Llama-3.1-8B (DP2)
- [ ] SFT warmup on translated AdvBench (DP3)
- [ ] GRPO main run (DP4)
- [ ] Final eval + plots + submission (DP5)
