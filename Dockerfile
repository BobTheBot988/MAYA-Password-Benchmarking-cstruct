# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1

# Choose PyTorch channel at build time: cu121 (default) or cpu
ARG TORCH_CHANNEL=cu121
# If your requirements.txt includes torch/vision/audio, this makes pip pull the right wheels
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/${TORCH_CHANNEL}

WORKDIR /app

# Common runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install deps (prefer wheels)
COPY requirements.txt .
RUN python -m pip install -U pip wheel \
  && pip install --prefer-binary -r requirements.txt


# App code
COPY . .

# Non-root
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Healthcheck: OK on CPU-only and on GPU
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python - <<'PY'\nimport sys, torch\nok = True\ntry:\n    _ = torch.ones(1)  # CPU op\n    if torch.cuda.is_available():\n        _ = torch.ones(1, device='cuda')\nexcept Exception:\n    ok = False\nsys.exit(0 if ok else 1)\nPY

# EXPOSE 8000
CMD ["python", "main.py"]

