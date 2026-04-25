#!/usr/bin/env bash
# Launch GRPO training on HF Jobs.
# Usage: bash scripts/run_grpo_job.sh [flavor]
# Default: t4-small ($0.60/hr). For faster: a10g-small, l4x1
set -euo pipefail

FLAVOR="${1:-t4-small}"

echo "→ launching GRPO job on $FLAVOR"

cd "$(dirname "$0")/.."

unset HF_TOKEN  # avoid invalid .env token overriding cached login

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
    secrets={'HF_TOKEN': token},
    env={
        'MOCK_GPU': '0',
        'GRPO_HUB_REPO': 'Saiyam0211/polyglot-redteam-grpo',
        'SFT_ADAPTER': 'Saiyam0211/polyglot-redteam-sft',
        'SPACE_URL': 'https://saiyam0211-polyglot-redteam.hf.space',
    },
    token=token,
)
print(f'Job ID: {job.id}')
print(f'Status: {job.status}')
print(f'URL: https://huggingface.co/jobs/Saiyam0211/{job.id}')
"
