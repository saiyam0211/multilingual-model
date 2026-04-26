#!/usr/bin/env bash
# Eval v3 job — run on HF Jobs (T4 sufficient).
set -euo pipefail

echo "=== Eval v3 Job ==="

pip install --quiet --upgrade pip
pip install --quiet torch transformers peft accelerate datasets trl bitsandbytes
pip install --quiet unsloth sentence-transformers
pip install --quiet fastapi httpx tenacity pydantic pydantic-settings structlog
pip install --quiet deep_translator fasttext-wheel
pip install --quiet huggingface_hub numpy

pip install --quiet -e ".[gpu]"

export MOCK_GPU=0
export EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-Saiyam0211/polyglot-redteam-grpo-v3}
export EVAL_DATA=${EVAL_DATA:-data/eval_prompts_v3.jsonl}
export EVAL_NUM_GENERATIONS=${EVAL_NUM_GENERATIONS:-3}

python scripts/eval_v3.py

echo "✓ Eval v3 job complete"
