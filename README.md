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

## Docker

Build and run the FastAPI service locally:

```bash
docker build -t football-pred-api .
docker run --rm -p 8080:8080 football-pred-api
```

## Continuous deployment to Google Cloud Run

A GitHub Actions workflow is available at `.github/workflows/deploy-cloud-run.yml`.
On each push to `master`, it builds a container image, pushes it to Artifact Registry, and deploys it to Cloud Run.

Required repository secrets:

- `GCP_PROJECT_ID`: Google Cloud project ID.
- `GCP_REGION`: Region for Artifact Registry and Cloud Run (example: `europe-west1`).
- `GCP_WORKLOAD_IDENTITY_PROVIDER`: Workload Identity Provider resource name.
- `GCP_SERVICE_ACCOUNT`: Service account email used by GitHub Actions.
