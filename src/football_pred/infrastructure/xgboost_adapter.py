from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
from returns.result import Failure, Result, Success
from xgboost import XGBClassifier

from football_pred.application.ports import TrainingSummary
from football_pred.domain.entities import MatchFeatures, MatchOutcome, MatchPrediction

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_form_5",
    "away_form_5",
    "home_goals_for_avg",
    "away_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_against_avg",
    "home_rest_days",
    "away_rest_days",
    "home_h2h_win_rate",
    "away_h2h_win_rate",
    "home_gd_form",
    "away_gd_form",
)
TARGET_MAPPING: Final[dict[str, int]] = {"H": 0, "D": 1, "A": 2}
REVERSE_TARGET_MAPPING: Final[dict[int, MatchOutcome]] = {
    0: MatchOutcome.HOME,
    1: MatchOutcome.DRAW,
    2: MatchOutcome.AWAY,
}


@dataclass(frozen=True)
class XGBoostTrainingAdapter:
    random_state: int = 42

    def train_model(self, dataset_path: str, model_output_path: str) -> Result[TrainingSummary, str]:
        try:
            dataframe = pd.read_csv(dataset_path)
            if dataframe.empty:
                return Failure("Dataset is empty")
            if "result" not in dataframe.columns:
                return Failure("Dataset must contain a 'result' column")

            x_values = dataframe.loc[:, FEATURE_COLUMNS]
            y_values = dataframe["result"].map(TARGET_MAPPING)
            if y_values.isna().any():
                return Failure("Target column contains values outside H/D/A")

            model = XGBClassifier(
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=self.random_state,
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
            )
            model.fit(x_values, y_values)

            output_path = Path(model_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as model_file:
                pickle.dump(model, model_file)

            return Success(TrainingSummary(rows=len(dataframe), model_path=str(output_path)))
        except Exception as exc:  # infrastructure boundary
            return Failure(f"Training failed: {exc}")


@dataclass(frozen=True)
class XGBoostPredictionAdapter:
    model: XGBClassifier

    @classmethod
    def from_pickle(cls, model_path: str) -> Result["XGBoostPredictionAdapter", str]:
        try:
            with Path(model_path).open("rb") as model_file:
                model = pickle.load(model_file)
            return Success(cls(model=model))
        except Exception as exc:  # infrastructure boundary
            return Failure(f"Loading model failed: {exc}")

    def predict(self, features: MatchFeatures) -> Result[MatchPrediction, str]:
        try:
            row = {
                "home_elo": features.home_elo,
                "away_elo": features.away_elo,
                "elo_diff": features.elo_diff,
                "home_form_5": features.home_form_5,
                "away_form_5": features.away_form_5,
                "home_goals_for_avg": features.home_goals_for_avg,
                "away_goals_for_avg": features.away_goals_for_avg,
                "home_goals_against_avg": features.home_goals_against_avg,
                "away_goals_against_avg": features.away_goals_against_avg,
                "home_rest_days": features.home_rest_days,
                "away_rest_days": features.away_rest_days,
                "home_h2h_win_rate": features.home_h2h_win_rate,
                "away_h2h_win_rate": features.away_h2h_win_rate,
                "home_gd_form": features.home_gd_form,
                "away_gd_form": features.away_gd_form,
            }
            dataframe = pd.DataFrame([row], columns=list(FEATURE_COLUMNS))
            probabilities = self.model.predict_proba(dataframe)[0]
            predicted_index = int(probabilities.argmax())
            prediction = MatchPrediction(
                predicted_outcome=REVERSE_TARGET_MAPPING[predicted_index],
                home_probability=float(probabilities[0]),
                draw_probability=float(probabilities[1]),
                away_probability=float(probabilities[2]),
            )
            return Success(prediction)
        except Exception as exc:  # infrastructure boundary
            return Failure(f"Prediction failed: {exc}")
