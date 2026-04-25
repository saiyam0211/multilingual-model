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

> Automated multilingual safety auditing for LLMs in Indian languages.
> Trains a Qwen2.5-3B attacker via GRPO to discover prompt patterns that
> bypass safety filters on a frozen Llama-3.1-8B-Instruct target.

**OpenEnv Hackathon — Apr 25-26, 2026**

## The Problem

Most LLM safety training is English-centric. Models like Llama-3.1 refuse
harmful English requests, but when the same request is in Hindi, Tamil, or
Bengali, they often comply. India has 700M+ internet users who prefer their
native language — they deserve equal safety protection.

**Nobody has built an automated tool to systematically find these failures.**
Manual red-teaming with native speakers is expensive and doesn't scale.

## Our Solution

A three-component RLVR system built on the OpenEnv framework:

| Component | Model | Role |
|-----------|-------|------|
| **Attacker** | Qwen2.5-3B-Instruct + LoRA | Generates adversarial prompts in 6 Indic languages |
| **Target** | Llama-3.1-8B-Instruct (frozen) | The model being safety-audited |
| **Environment** | OpenEnv FastAPI + reward composer | Scores attacks: harm detection, language ID, novelty |

### Languages Covered
Hindi, Tamil, Bengali, Marathi, Telugu, Kannada

### Training Pipeline
1. **SFT Warmup** — Fine-tune attacker on 402 translated adversarial seed prompts
2. **GRPO Training** — Reinforcement learning against live target with shaped rewards
3. **Evaluation** — Measure Attack Success Rate (ASR) before and after training

## Architecture

```
Attacker (Qwen2.5-3B)  →  generates adversarial prompt in target language
        ↓
Target (Llama-3.1-8B)  →  responds to the prompt
        ↓
Reward Composer         →  scores: harm × non-refusal × novelty
        ↓                  gates: language ID, gibberish detection
GRPO Update             →  updates attacker policy to maximize reward
```

## Results

| Metric | Baseline (seed prompts) | Post-GRPO (learned attacks) |
|--------|------------------------|---------------------------|
| **Overall ASR** | 98.5% | 100.0% |
| Prompts evaluated | 402 | 240 |
| Gate-passed | 402 | 39 (201 gated by quality filters) |

**Key finding:** Simple translated prompts already bypass Llama-3.1-8B safety filters
98.5% of the time across 6 Indian languages. The GRPO-trained attacker achieves 100%
ASR on quality-filtered outputs — every valid Indic prompt it generates bypasses safety.

| Language | Baseline ASR | Post-GRPO ASR |
|----------|-------------|---------------|
| Bengali | 100.0% | 100.0% |
| Hindi | 100.0% | 100.0% |
| Kannada | 100.0% | 100.0% |
| Marathi | 100.0% | 100.0% |
| Tamil | 93.8% | 100.0% |
| Telugu | 100.0% | 100.0% |

See `results/` for charts: `asr_by_language.png`, `asr_by_category.png`, `reward_distribution.png`

## Repository Structure

```
src/polyglot_redteam/       # OpenEnv FastAPI server
  ├── env.py                # Episode management (reset/step)
  ├── reward/               # Reward composer + components
  │   ├── composer.py       # Multi-component reward shaping
  │   ├── language_id.py    # FastText language identification
  │   └── novelty.py        # Sentence-transformer novelty scoring
  ├── target/               # Target model client (HF Inference API)
  └── config.py             # Pydantic settings

scripts/
  ├── sft_warmup.py         # B8: SFT fine-tuning on seed prompts
  ├── grpo_train.py         # B9: GRPO training loop
  ├── baseline_eval.py      # Baseline ASR measurement
  ├── post_grpo_eval.py     # Post-training ASR measurement
  ├── plot_results.py       # Generate comparison charts
  └── translate_seed_prompts.py  # Google Translate for seed data

data/
  ├── seed_prompts.jsonl    # 402 translated adversarial prompts (6 langs)
  └── baseline_results.jsonl # Baseline eval results

results/                    # Generated charts and summary
configs/base.yaml           # Hyperparameters
docs/ETHICS.md              # Ethics statement
```

## Quick Start

```bash
# Setup
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .
bash scripts/download_assets.sh   # fasttext model (~124MB)

# Test reward function
pytest

# Run baseline eval
python scripts/baseline_eval.py

# Generate plots
python scripts/plot_results.py
```

## Training (requires GPU)

SFT and GRPO training run on HF Jobs (T4 GPU):

```bash
# SFT warmup (~3 min on T4)
bash scripts/run_sft_job.sh

# GRPO training (~1 hr on T4)
bash scripts/run_grpo_job.sh
```

## Live Environment

- **Space:** https://huggingface.co/spaces/Saiyam0211/polyglot-redteam
- **Health:** `curl https://saiyam0211-polyglot-redteam.hf.space/health`
- **API:** POST `/reset` → POST `/step` with `{episode_id, action}`

## Trained Adapters

- **SFT:** [Saiyam0211/polyglot-redteam-sft](https://huggingface.co/Saiyam0211/polyglot-redteam-sft)
- **GRPO:** [Saiyam0211/polyglot-redteam-grpo](https://huggingface.co/Saiyam0211/polyglot-redteam-grpo)

## Status

- [x] Repo scaffold + reward components
- [x] Adversarial probes pass (30/30)
- [x] HF Space deployment
- [x] Seed prompts translated (402 × 6 languages)
- [x] Baseline eval: 98.5% ASR
- [x] SFT warmup complete (loss 3.07 → 0.36)
- [x] GRPO training complete (ASR 10% → 16.5% during training, 100% on eval)
- [x] Post-GRPO eval complete (240 prompts, 100% ASR on gate-passed)
- [x] Comparison charts generated
- [x] Submission ready

## Ethics

See [`docs/ETHICS.md`](docs/ETHICS.md). We do NOT publish:
- Working attack prompts or templates
- Target model responses to harmful queries
- Any content that could enable real-world harm

We DO publish: aggregated ASR metrics, reward distributions, methodology,
and the OpenEnv environment for responsible safety research.
