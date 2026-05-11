FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY app ./app
COPY README.md ./README.md

RUN mkdir -p /app/storage \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "HOST=\"${API_HOST:-0.0.0.0}\"; PORT=\"${PORT:-${API_PORT:-8000}}\"; exec uvicorn app.main:app --host \"$HOST\" --port \"$PORT\""]
