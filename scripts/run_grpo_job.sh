#!/usr/bin/env bash
# Launch GRPO training on HF Jobs.
# Usage: bash scripts/run_grpo_job.sh [flavor]
# Default flavor: l4x1 ($0.80/hr). Alternatives: a10g-small ($1/hr), a100-large ($2.50/hr)
set -euo pipefail

FLAVOR="${1:-l4x1}"

echo "→ launching GRPO job on $FLAVOR"

cd "$(dirname "$0")/.."
source .venv/bin/activate

python -c "
from huggingface_hub import HfApi, get_token
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
    flavor='$FLAVOR',
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
print(f'URL: https://huggingface.co/jobs/Saiyam0211/{job.id}')
print(f'Monitor: hf jobs logs {job.id}')
"
