#!/usr/bin/env bash
# Launch GRPO v2 training on top of SFT v2 to increase raw zero-shot yield.
set -euo pipefail

FLAVOR="${1:-t4-small}"

echo "→ launching GRPO v2 job on $FLAVOR"
cd "$(dirname "$0")/.."
unset HF_TOKEN

python3 -c "
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

api = HfApi()
token = get_token()

job = api.run_job(
    image='ghcr.io/unslothai/unsloth:latest',
    command=['bash', '-c',
        'cd /tmp && git clone https://github.com/saiyam0211/multilingual-model.git repo && cd repo '
        '&& pip install -q deep-translator structlog pydantic-settings tenacity python-dotenv httpx '
        'fastapi uvicorn sentence-transformers fasttext-wheel huggingface_hub '
        '&& pip install -q -e . && bash scripts/download_assets.sh '
        '&& python scripts/grpo_train.py'
    ],
    flavor='${FLAVOR}',
    timeout='6h',
    secrets={'HF_TOKEN': token},
    env={
        'MOCK_GPU': '0',
        'SPACE_URL': 'https://saiyam0211-polyglot-redteam.hf.space/api',
        'SFT_ADAPTER': 'Saiyam0211/polyglot-redteam-sft-v2',
        'GRPO_HUB_REPO': 'Saiyam0211/polyglot-redteam-grpo-v2',
        'GRPO_OUTPUT_DIR': 'checkpoints/grpo-v2',
        'GRPO_RUN_NAME': 'grpo-v2-yield-boost',
        'GRPO_MAX_STEPS': '320',
        'GRPO_NUM_GENERATIONS': '6',
        'GRPO_GRAD_ACCUM': '4',
        'GRPO_BATCH_SIZE': '1',
        'GRPO_LR': '3e-6',
        'GRPO_WARMUP_RATIO': '0.08',
        'GRPO_SAVE_EVERY': '80',
        'GRPO_TEMPERATURE': '0.95',
    },
    token=token,
)
print(f'Job ID: {job.id}')
print(f'Status: {job.status}')
print(f'URL: https://huggingface.co/jobs/Saiyam0211/{job.id}')
"

