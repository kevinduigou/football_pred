FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY xgb_football_model.json ./xgb_football_model.json

RUN pip install --upgrade pip \
    && pip install .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn football_pred.interface.fastapi_app:app --host 0.0.0.0 --port ${PORT:-8080}"]
