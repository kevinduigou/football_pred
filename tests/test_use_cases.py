from __future__ import annotations

from dataclasses import dataclass

from returns.result import Failure, Result, Success

from football_pred.application.ports import TrainingSummary
from football_pred.application.use_cases import PredictMatchUseCase, TrainModelUseCase
from football_pred.domain.entities import MatchFeatures, MatchOutcome, MatchPrediction


@dataclass(frozen=True)
class SuccessfulTrainer:
    def train_model(self, dataset_path: str, model_output_path: str) -> Result[TrainingSummary, str]:
        return Success(TrainingSummary(rows=10, model_path=model_output_path))


@dataclass(frozen=True)
class SuccessfulPredictor:
    def predict(self, features: MatchFeatures) -> Result[MatchPrediction, str]:
        return Success(
            MatchPrediction(
                predicted_outcome=MatchOutcome.HOME,
                home_probability=0.6,
                draw_probability=0.2,
                away_probability=0.2,
            )
        )


def test_train_model_use_case_returns_summary() -> None:
    use_case = TrainModelUseCase(trainer=SuccessfulTrainer())
    result = use_case.execute(dataset_path="input.csv", model_output_path="model.pkl")

    assert isinstance(result, Success)
    assert result.unwrap().model_path == "model.pkl"


def test_predict_match_use_case_rejects_missing_feature() -> None:
    use_case = PredictMatchUseCase(predictor=SuccessfulPredictor())
    result = use_case.execute(raw_features={"home_elo": 1500.0})

    assert isinstance(result, Failure)


def test_predict_match_use_case_returns_prediction() -> None:
    use_case = PredictMatchUseCase(predictor=SuccessfulPredictor())
    features = {
        "home_elo": 1620.0,
        "away_elo": 1580.0,
        "elo_diff": 40.0,
        "home_form_5": 2.0,
        "away_form_5": 1.2,
        "home_goals_for_avg": 1.8,
        "away_goals_for_avg": 1.3,
        "home_goals_against_avg": 0.9,
        "away_goals_against_avg": 1.1,
        "home_rest_days": 6.0,
        "away_rest_days": 4.0,
        "home_h2h_win_rate": 0.5,
        "away_h2h_win_rate": 0.25,
        "home_gd_form": 0.9,
        "away_gd_form": 0.2,
    }

    result = use_case.execute(raw_features=features)

    assert isinstance(result, Success)
    assert result.unwrap().predicted_outcome is MatchOutcome.HOME
