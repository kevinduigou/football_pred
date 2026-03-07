from __future__ import annotations

import typer
import uvicorn
from returns.result import Failure

from football_pred.application.use_cases import TrainModelUseCase
from football_pred.infrastructure.xgboost_adapter import XGBoostTrainingAdapter

cli_app = typer.Typer(help="Football prediction CLI")


@cli_app.command("train")
def train_model(
    dataset_path: str = typer.Option("football_matches.csv", help="Path to training CSV"),
    model_output_path: str = typer.Option("xgb_football_model.pkl", help="Output model file"),
) -> None:
    use_case = TrainModelUseCase(trainer=XGBoostTrainingAdapter())
    training_result = use_case.execute(dataset_path=dataset_path, model_output_path=model_output_path)

    if isinstance(training_result, Failure):
        typer.echo(f"Training failed: {training_result.failure()}")
        raise typer.Exit(code=1)

    summary = training_result.unwrap()
    typer.echo(f"Training complete on {summary.rows} rows")
    typer.echo(f"Model saved to {summary.model_path}")


@cli_app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", help="FastAPI host"),
    port: int = typer.Option(8000, help="FastAPI port"),
) -> None:
    uvicorn.run("football_pred.interface.fastapi_app:app", host=host, port=port, reload=False)


def main() -> None:
    cli_app()
