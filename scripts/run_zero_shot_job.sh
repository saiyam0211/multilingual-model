#!/usr/bin/env bash
# Launch zero-shot generation proof job on HF Jobs.
set -euo pipefail

FLAVOR="${1:-t4-small}"

echo "→ launching zero-shot generation job on $FLAVOR"

cd "$(dirname "$0")/.."

unset HF_TOKEN  # use cached HF login

python3 -c "
from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

api = HfApi()
token = get_token()

job = api.run_job(
    image='ghcr.io/unslothai/unsloth:latest',
    command=['bash', '-c',
        'cd /tmp && git clone https://github.com/saiyam0211/multilingual-model.git repo && cd repo '
        '&& pip install -q deep-translator huggingface_hub peft transformers '
        '&& python scripts/zero_shot_generation.py'
    ],
    flavor='${FLAVOR}',
    secrets={'HF_TOKEN': token},
    env={'HF_HOME': '/tmp/hf_cache'},
    token=token,
)
print(f'Job ID: {job.id}')
print(f'Status: {job.status}')
print(f'URL: https://huggingface.co/jobs/Saiyam0211/{job.id}')
"
