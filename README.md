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
> We discovered **1094 confirmed safety-gap prompts** in Llama-3.1-8B-Instruct:
> requests the model **refuses in English but answers in Hindi, Tamil, Bengali,
> Marathi, Telugu, or Kannada**.

**OpenEnv Hackathon — Apr 25-26, 2026**

## 🎯 Headline Result

**1094 confirmed multilingual safety-gap prompts** in a curated dataset, published as
[`Saiyam0211/polyglot-redteam-vulnerabilities`](https://huggingface.co/datasets/Saiyam0211/polyglot-redteam-vulnerabilities) for responsible disclosure to model providers.

| | |
|---|---|
| Total prompts tested | 1,780 |
| Confirmed safety gaps (English-refused, Indic-answered) | **1,094** |
| Discovery rate | **61.5 %** |
| Languages covered | Hindi, Tamil, Bengali, Marathi, Telugu, Kannada |
| Harm categories | Illicit finance, privacy, scams, violence enablement |
| Attack wrappers | Direct, educational, hypothetical, persona, step-by-step, indirect, combo |

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

### The Safety Gap (English vs Indian Languages)

We sent the same harmful prompts to Llama-3.1-8B-Instruct in English and 6 Indian
languages. The model refuses 73% of English requests but **0% of Indic requests**.

| Language | Refusal Rate | Attack Success Rate |
|----------|-------------|-------------------|
| **English** | **73.3%** | **26.7%** |
| Hindi | 0.0% | 100.0% |
| Tamil | 0.0% | 100.0% |
| Bengali | 0.0% | 100.0% |
| Marathi | 0.0% | 100.0% |
| Telugu | 0.0% | 100.0% |
| Kannada | 0.0% | 100.0% |

**Safety gap: +75 percentage points.** Violence and privacy violations show the
worst gaps (+86pp each).

### Vulnerability Dataset (1094 confirmed gaps)

| Language | Confirmed Gaps |
|----------|---------------:|
| Kannada | 222 |
| Tamil | 202 |
| Bengali | 202 |
| Marathi | 185 |
| Telugu | 158 |
| Hindi | 125 |
| **Total** | **1,094** |

| Category | Confirmed Gaps |
|----------|---------------:|
| Privacy violation | 317 |
| Violence enablement | 274 |
| Illicit finance | 266 |
| Scam engineering | 237 |

### Pipeline → Dataset

1. **Seed translation** — 67 harmful English prompts → 6 languages = 402 baseline pairs.
   - 230 confirmed gaps (57.2 % discovery rate).
2. **GRPO training** — Qwen2.5-3B attacker trained on OpenEnv. Learns to generate
   novel adversarial Indic prompts (100 % ASR on gate-passed eval outputs).
3. **Augmentation** — 5 attack-style wrappers (educational, hypothetical, persona,
   step-by-step, indirect) applied multilingually to each confirmed gap.
   - +575 new gaps (1150 candidates → 50 % yield).
4. **Combo wrapper** — strongest two wrappers combined.
   - +289 new gaps (400 candidates → 72 % yield).
5. **Final curation** — every prompt in the published dataset is confirmed to
   trigger `English=refused AND Indic=answered` on Llama-3.1-8B-Instruct.

See `results/` for charts: `safety_gap_comparison.png`, `safety_gap_by_category.png`,
`findings_summary.png`, `asr_by_language.png`

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

## Trained Adapters & Dataset

- **SFT:** [Saiyam0211/polyglot-redteam-sft](https://huggingface.co/Saiyam0211/polyglot-redteam-sft)
- **GRPO:** [Saiyam0211/polyglot-redteam-grpo](https://huggingface.co/Saiyam0211/polyglot-redteam-grpo)
- **Vulnerability Dataset:** [Saiyam0211/polyglot-redteam-vulnerabilities](https://huggingface.co/datasets/Saiyam0211/polyglot-redteam-vulnerabilities)

## Use Case (Why this matters)

This is **not a defended model**. It is a **vulnerability scanner that produces
a curated dataset for responsible disclosure**.

**Who uses the dataset?**
- **Model providers** (Meta, Mistral, OpenAI, etc.) — to retrain safety on the
  exact prompts that bypass their current refusals.
- **Safety researchers** — to benchmark cross-lingual safety transfer.
- **Indic LLM teams** (Sarvam, Krutrim, BharatGen, AI4Bharat) — to validate
  that their models block what frontier models miss.

**Why automation matters:** Manual red-teaming with native speakers costs
~$50/prompt. Our pipeline generates and verifies 1094 confirmed gaps for
under $5 of inference compute.

## Status

- [x] Repo scaffold + reward components
- [x] Adversarial probes pass (30/30)
- [x] HF Space deployment with 4-tab Gradio UI (Browse · Live Test · Stats · About)
- [x] Seed prompts translated (402 × 6 languages)
- [x] Baseline eval: 98.5 % ASR
- [x] SFT warmup complete (loss 3.07 → 0.36)
- [x] GRPO training complete (ASR 10 % → 100 % on gate-passed eval)
- [x] **Vulnerability dataset built: 1094 confirmed safety gaps**
- [x] Dataset published to HF Hub
- [x] Submission ready

## Ethics

See [`docs/ETHICS.md`](docs/ETHICS.md). We do NOT publish:
- Working attack prompts or templates
- Target model responses to harmful queries
- Any content that could enable real-world harm

We DO publish: aggregated ASR metrics, reward distributions, methodology,
and the OpenEnv environment for responsible safety research.
