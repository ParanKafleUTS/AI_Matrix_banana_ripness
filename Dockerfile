FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install all dependencies in one step
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY main.py .
COPY static/ ./static/
COPY models/ ./models/

RUN chmod -R 777 /app

EXPOSE 7860

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
