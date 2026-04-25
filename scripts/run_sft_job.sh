#!/usr/bin/env bash
# Entrypoint for HF Jobs SFT warmup.
# Runs inside ghcr.io/unslothai/unsloth:latest container on GPU.
set -euo pipefail

echo "=== SFT warmup job starting ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Python: $(python --version)"

# Clone repo
cd /tmp
git clone https://github.com/saiyam0211/multilingual-model.git repo
cd repo

# Install our package + deps not in unsloth image
pip install -q deep-translator structlog pydantic-settings tenacity python-dotenv httpx fastapi uvicorn sentence-transformers fasttext-wheel huggingface_hub
pip install -q -e .

# Download fasttext lid model
bash scripts/download_assets.sh

# Run SFT
echo "=== starting SFT training ==="
python scripts/sft_warmup.py

echo "=== SFT job complete ==="
