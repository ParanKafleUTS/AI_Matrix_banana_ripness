# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY main.py .
COPY static/ static/

# Copy model weights (build with --build-arg or mount at runtime)
# The model is loaded lazily, so the container starts even without weights.
# To include weights at build time:  COPY saved_models/ saved_models/

ENV MODEL_PATH=saved_models/EfficientNetB0_banana_ripeness.pth

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
