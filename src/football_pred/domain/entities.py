from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Sequence

from returns.result import Failure, Result, Success


class MatchOutcome(str, Enum):
    HOME = "H"
    DRAW = "D"
    AWAY = "A"


@dataclass(frozen=True)
class MatchFeatures:
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

    @classmethod
    def try_create(cls, raw_features: Mapping[str, float]) -> Result["MatchFeatures", str]:
        required_keys: Sequence[str] = (
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

        missing_keys = [key for key in required_keys if key not in raw_features]
        if missing_keys:
            return Failure(f"Missing features: {', '.join(missing_keys)}")

        normalized_values: dict[str, float] = {}
        for key in required_keys:
            raw_value = raw_features[key]
            if isinstance(raw_value, bool) or not isinstance(raw_value, (float, int)):
                return Failure(f"Feature '{key}' must be numeric")

            numeric_value = float(raw_value)
            if not isfinite(numeric_value):
                return Failure(f"Feature '{key}' must be finite")
            if key.endswith("rest_days") and numeric_value < 0:
                return Failure(f"Feature '{key}' must be >= 0")
            normalized_values[key] = numeric_value

        return Success(
            cls(
                home_elo=normalized_values["home_elo"],
                away_elo=normalized_values["away_elo"],
                elo_diff=normalized_values["elo_diff"],
                home_form_5=normalized_values["home_form_5"],
                away_form_5=normalized_values["away_form_5"],
                home_goals_for_avg=normalized_values["home_goals_for_avg"],
                away_goals_for_avg=normalized_values["away_goals_for_avg"],
                home_goals_against_avg=normalized_values["home_goals_against_avg"],
                away_goals_against_avg=normalized_values["away_goals_against_avg"],
                home_rest_days=normalized_values["home_rest_days"],
                away_rest_days=normalized_values["away_rest_days"],
                home_h2h_win_rate=normalized_values["home_h2h_win_rate"],
                away_h2h_win_rate=normalized_values["away_h2h_win_rate"],
                home_gd_form=normalized_values["home_gd_form"],
                away_gd_form=normalized_values["away_gd_form"],
            )
        )


@dataclass(frozen=True)
class MatchPrediction:
    predicted_outcome: MatchOutcome
    home_probability: float
    draw_probability: float
    away_probability: float
