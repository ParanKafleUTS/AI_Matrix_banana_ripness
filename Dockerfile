# ── Banana Ripeness API — Dockerfile ──────────────────────────────────────────
# Compatible with:
#   - Hugging Face Spaces (exposes port 7860)
#   - Railway / Render    (reads $PORT env var)
#   - Local dev           (defaults to 8000)

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY main.py .
COPY static/ ./static/

# HF Spaces runs as a non-root user — pre-create writable dirs
RUN mkdir -p models && chmod -R 777 models && chmod -R 777 /app

EXPOSE 7860

# HF Spaces sets PORT=7860; Railway sets its own $PORT; local defaults to 8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
