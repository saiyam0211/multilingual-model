#!/usr/bin/env bash
# SFT v3 training job — run on HF Jobs.
set -euo pipefail

echo "=== SFT v3 Training Job ==="

pip install --quiet --upgrade pip
pip install --quiet torch transformers peft accelerate datasets trl bitsandbytes
pip install --quiet unsloth sentence-transformers
pip install --quiet pydantic pydantic-settings structlog python-dotenv
pip install --quiet wandb huggingface_hub

pip install --quiet -e ".[gpu]"

export MOCK_GPU=0
export SFT_BASE_MODEL=${SFT_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}
export SFT_DATA=${SFT_DATA:-data/sft_v3.jsonl}
export SFT_OUTPUT_DIR=${SFT_OUTPUT_DIR:-checkpoints/sft_v3}
export SFT_HUB_REPO=${SFT_HUB_REPO:-Saiyam0211/polyglot-redteam-sft-v3}

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT="MultiLingual Model"
    export WANDB_ENTITY="${WANDB_ENTITY:-saiyam0211-defellix}"
fi

# Build dataset first
python scripts/build_sft_v3.py

# Train
python scripts/sft_v3.py

echo "✓ SFT v3 job complete"
