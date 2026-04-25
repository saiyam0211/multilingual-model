FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf_cache \
    TRANSFORMERS_CACHE=/data/hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src
RUN pip install -U pip && pip install -e .

# fasttext lid model (124MB) — bake into image so cold-start is fast
RUN mkdir -p /app/data && \
    curl -L -o /app/data/lid.176.bin \
    https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

EXPOSE 8000

CMD ["uvicorn", "polyglot_redteam.server:app", "--host", "0.0.0.0", "--port", "8000"]
