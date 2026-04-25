#!/usr/bin/env bash
# Launch SFT v2 (yield-boost curriculum) on HF Jobs.
set -euo pipefail

FLAVOR="${1:-t4-small}"

echo "→ launching SFT v2 job on $FLAVOR"

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
        '&& pip install -q huggingface_hub datasets trl peft transformers '
        '&& python scripts/sft_attacker_v2.py'
    ],
    flavor='${FLAVOR}',
    timeout='2h',
    secrets={'HF_TOKEN': token},
    env={
        'HF_HOME': '/tmp/hf_cache',
        'SFT_V2_HUB_REPO': 'Saiyam0211/polyglot-redteam-sft-v2',
        'SFT_V2_BASE': 'Saiyam0211/polyglot-redteam-grpo',
        'SFT_V2_EPOCHS': '3',
        'SFT_V2_LR': '1.5e-4',
    },
    token=token,
)
print(f'Job ID: {job.id}')
print(f'Status: {job.status}')
print(f'URL: https://huggingface.co/jobs/Saiyam0211/{job.id}')
"

