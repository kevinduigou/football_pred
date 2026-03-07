# football-pred

Football prediction project reorganized with **Hexagonal Architecture**:

- `domain`: immutable business entities and validation.
- `application`: use cases and ports.
- `infrastructure`: XGBoost adapters for training and prediction.
- `interface`: Typer CLI and FastAPI HTTP API.

## Train model (Typer CLI)

```bash
uv run football-pred train --dataset-path football_matches.csv --model-output-path xgb_football_model.pkl
```

## Run API (FastAPI)

```bash
uv run football-pred serve --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /predict`
