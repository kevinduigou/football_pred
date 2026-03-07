from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from returns.result import Failure, Result, Success

from football_pred.application.ports import ModelPredictionPort, ModelTrainingPort, TrainingSummary
from football_pred.domain.entities import MatchPrediction, MatchFeatures


@dataclass(frozen=True)
class TrainModelUseCase:
    trainer: ModelTrainingPort

    def execute(self, dataset_path: str, model_output_path: str) -> Result[TrainingSummary, str]:
        return self.trainer.train_model(dataset_path=dataset_path, model_output_path=model_output_path)


@dataclass(frozen=True)
class PredictMatchUseCase:
    predictor: ModelPredictionPort

    def execute(self, raw_features: Mapping[str, float]) -> Result[MatchPrediction, str]:
        features_result = MatchFeatures.try_create(raw_features=raw_features)
        if isinstance(features_result, Failure):
            return Failure(features_result.failure())

        features = features_result.unwrap()
        prediction_result = self.predictor.predict(features=features)
        if isinstance(prediction_result, Failure):
            return Failure(prediction_result.failure())

        return Success(prediction_result.unwrap())
