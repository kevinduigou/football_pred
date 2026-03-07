from __future__ import annotations

from os import getenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from returns.result import Failure

from football_pred.application.use_cases import PredictMatchUseCase
from football_pred.infrastructure.xgboost_adapter import XGBoostPredictionAdapter


class MatchFeatureRequest(BaseModel):
    home_elo: float
    away_elo: float
    elo_diff: float
    home_form_5: float
    away_form_5: float
    home_goals_for_avg: float
    away_goals_for_avg: float
    home_goals_against_avg: float
    away_goals_against_avg: float
    home_rest_days: float
    away_rest_days: float
    home_h2h_win_rate: float
    away_h2h_win_rate: float
    home_gd_form: float
    away_gd_form: float


def create_app(model_path: str | None = None) -> FastAPI:
    resolved_model_path = model_path or getenv("MODEL_PATH", "xgb_football_model.pkl")
    adapter_result = XGBoostPredictionAdapter.from_pickle(model_path=resolved_model_path)

    app = FastAPI(title="Football Prediction API", version="1.0.0")

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        status = "ready" if not isinstance(adapter_result, Failure) else "model_not_loaded"
        return {"status": status}

    @app.post("/predict")
    def predict(request: MatchFeatureRequest) -> dict[str, float | str]:
        if isinstance(adapter_result, Failure):
            raise HTTPException(status_code=500, detail=adapter_result.failure())

        predictor_use_case = PredictMatchUseCase(predictor=adapter_result.unwrap())
        prediction_result = predictor_use_case.execute(raw_features=request.model_dump())

        if isinstance(prediction_result, Failure):
            raise HTTPException(status_code=400, detail=prediction_result.failure())

        prediction = prediction_result.unwrap()
        return {
            "predicted_outcome": prediction.predicted_outcome.value,
            "home_probability": prediction.home_probability,
            "draw_probability": prediction.draw_probability,
            "away_probability": prediction.away_probability,
        }

    return app


app = create_app()
