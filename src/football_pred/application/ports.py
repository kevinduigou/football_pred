from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from returns.result import Result

from football_pred.domain.entities import MatchFeatures, MatchPrediction


@dataclass(frozen=True)
class TrainingSummary:
    rows: int
    model_path: str


class ModelTrainingPort(Protocol):
    def train_model(self, dataset_path: str, model_output_path: str) -> Result[TrainingSummary, str]:
        ...


class ModelPredictionPort(Protocol):
    def predict(self, features: MatchFeatures) -> Result[MatchPrediction, str]:
        ...
