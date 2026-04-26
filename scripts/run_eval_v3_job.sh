#!/usr/bin/env bash
# Eval v3 job — run on HF Jobs (L40S recommended for Aya local judge).
#
# Usage:
#   hf jobs run --flavor l40sx1 --secrets HF_TOKEN \
#     -d ghcr.io/unslothai/unsloth:latest \
#     bash -c "git clone https://github.com/saiyam0211/multilingual-model.git && \
#              cd multilingual-model && bash scripts/run_eval_v3_job.sh"

set -euo pipefail

echo "=== Eval v3 Job ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

# ---- Install deps -----------------------------------------------------------
pip install --quiet --upgrade pip
pip install --quiet torch transformers peft accelerate datasets trl bitsandbytes
pip install --quiet unsloth vllm sentence-transformers
pip install --quiet fastapi httpx tenacity pydantic pydantic-settings structlog
pip install --quiet deep_translator fasttext-wheel gradio
pip install --quiet huggingface_hub numpy pyyaml wandb

# ---- Install project --------------------------------------------------------
pip install --quiet -e ".[gpu]"

# ---- Download fasttext language ID model ------------------------------------
LID_MODEL="models/lid.176.bin"
if [ ! -f "$LID_MODEL" ]; then
    echo "→ Downloading fasttext lid model..."
    mkdir -p models
    python -c "
from huggingface_hub import hf_hub_download
import os, shutil
path = hf_hub_download('facebook/fasttext-language-identification', 'model.bin', token=os.environ.get('HF_TOKEN'))
shutil.copy(path, 'models/lid.176.bin')
print('  ✓ lid.176.bin downloaded')
" || echo "  ⚠ fasttext download failed — will use script-based fallback"
fi

# ---- Environment ------------------------------------------------------------
export MOCK_GPU=0
export EVAL_CHECKPOINT=${EVAL_CHECKPOINT:-Saiyam0211/polyglot-redteam-grpo-v3}
export EVAL_DATA=${EVAL_DATA:-data/eval_prompts_v3.jsonl}
export EVAL_NUM_GENERATIONS=${EVAL_NUM_GENERATIONS:-3}
export EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-194}

echo ""
echo "→ Eval config:"
echo "  Checkpoint: ${EVAL_CHECKPOINT}"
echo "  Eval data: ${EVAL_DATA}"
echo "  Generations: ${EVAL_NUM_GENERATIONS}"
echo "  Max samples: ${EVAL_MAX_SAMPLES}"

# Safety net: rebuild eval data if missing
if [ ! -f "$EVAL_DATA" ]; then
    echo "⚠ Eval data missing — rebuilding from build_sft_v3.py..."
    python scripts/build_sft_v3.py
fi

# ---- Run eval ---------------------------------------------------------------
python scripts/eval_v3.py

# ---- Generate manual audit CSV ----------------------------------------------
echo ""
echo "→ Generating manual audit template..."
python scripts/manual_audit_template.py || echo "  ⚠ Audit template generation failed (non-critical)"

# ---- Upload results to HF Hub -----------------------------------------------
echo ""
echo "→ Uploading results..."
python -c "
from huggingface_hub import upload_folder
import os
upload_folder(
    folder_path='results',
    repo_id='Saiyam0211/polyglot-redteam-grpo-v3',
    path_in_repo='eval_results',
    token=os.environ.get('HF_TOKEN'),
)
print('  ✓ Results uploaded to HF Hub')
" || echo "  ⚠ Upload failed (non-critical) — results saved locally"

echo ""
echo "✓ Eval v3 job complete"
