#!/usr/bin/env bash
# GRPO v3 training job — run on HF Jobs (L4 recommended, T4-medium fallback).
#
# Usage:
#   export HF_TOKEN=hf_...
#   export WANDB_API_KEY=...
#   bash scripts/run_grpo_v3_job.sh
#
# Or via HF Jobs:
#   huggingface-cli job run \
#     --script scripts/run_grpo_v3_job.sh \
#     --hardware nvidia-l4 \
#     --repo-name Saiyam0211/polyglot-redteam-grpo-v3

set -euo pipefail

echo "=== GRPO v3 Training Job ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'none')"

# ---- Install deps -----------------------------------------------------------
pip install --quiet --upgrade pip
pip install --quiet torch transformers peft accelerate datasets trl bitsandbytes
pip install --quiet unsloth vllm sentence-transformers
pip install --quiet fastapi httpx tenacity pydantic pydantic-settings structlog
pip install --quiet deep_translator fasttext-wheel gradio
pip install --quiet wandb huggingface_hub pyyaml

# ---- Install project --------------------------------------------------------
pip install --quiet -e ".[gpu]"

# ---- Environment ------------------------------------------------------------
export MOCK_GPU=0
export GRPO_MAX_STEPS=${GRPO_MAX_STEPS:-1500}
export GRPO_NUM_GENERATIONS=${GRPO_NUM_GENERATIONS:-8}
export GRPO_BATCH_SIZE=${GRPO_BATCH_SIZE:-1}
export GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM:-8}
export GRPO_LR=${GRPO_LR:-1e-6}
export GRPO_TEMPERATURE=${GRPO_TEMPERATURE:-1.0}
export SFT_ADAPTER=${SFT_ADAPTER:-Saiyam0211/polyglot-redteam-sft-v3}
export GRPO_HUB_REPO=${GRPO_HUB_REPO:-Saiyam0211/polyglot-redteam-grpo-v3}
export GRPO_OUTPUT_DIR=${GRPO_OUTPUT_DIR:-checkpoints/grpo_v3}
export GRPO_CONFIG=configs/training/grpo_v3.yaml

# ---- W&B setup --------------------------------------------------------------
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT="MultiLingual Model"
    export WANDB_ENTITY="${WANDB_ENTITY:-saiyam0211-defellix}"
    echo "W&B enabled: ${WANDB_ENTITY}/${WANDB_PROJECT}"
fi

# ---- Run training -----------------------------------------------------------
echo ""
echo "→ Starting GRPO v3..."
echo "  SFT: ${SFT_ADAPTER}"
echo "  Steps: ${GRPO_MAX_STEPS}"
echo "  Generations: ${GRPO_NUM_GENERATIONS}"
echo "  LR: ${GRPO_LR}"
echo ""

python scripts/grpo_v3_train.py

echo ""
echo "✓ GRPO v3 job complete"
