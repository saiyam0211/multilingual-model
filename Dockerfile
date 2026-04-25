# HF Spaces Dockerfile — Gradio UI + OpenEnv FastAPI on free CPU hardware.
# Llama-Guard is mocked in-Space (MOCK_GPU=1); real harm classification happens
# during training on HF Jobs, not here.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf_cache \
    HF_HUB_CACHE=/data/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/data/hf_cache \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
COPY data/comparison_summary.json ./data/comparison_summary.json
COPY data/vulnerability_dataset_final.jsonl ./data/vulnerability_dataset_final.jsonl
RUN pip install -U pip && pip install --no-cache-dir -e . \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir gradio deep-translator

RUN mkdir -p /app/data && \
    curl -L -o /app/data/lid.176.bin \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

EXPOSE 7860

CMD ["sh", "-c", "uvicorn polyglot_redteam.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
