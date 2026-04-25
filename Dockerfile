# HF Spaces Dockerfile — runs the OpenEnv FastAPI server on free CPU hardware.
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
RUN pip install -U pip && pip install --no-cache-dir -e . \
 # CPU-only torch keeps the image small and the cold-start fast.
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Bake fasttext lid model (124MB) into the image — cold-start friendly.
RUN mkdir -p /app/data && \
    curl -L -o /app/data/lid.176.bin \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# HF Spaces requires the app to listen on $PORT (default 7860).
EXPOSE 7860

CMD ["sh", "-c", "uvicorn polyglot_redteam.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
