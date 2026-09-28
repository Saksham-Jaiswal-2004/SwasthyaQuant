from __future__ import annotations

from typing import Any

from app.schemas.patient import PatientInput
from app.schemas.prediction import PredictionResponse
from app.services.model_service import model_service
from app.services.preprocessing_service import PreprocessingService


class PredictionService:
    def __init__(self) -> None:
        self.preprocessing = PreprocessingService()

    def predict(self, patient: PatientInput) -> PredictionResponse:
        try:
            bundle = model_service.load()
        except Exception as exc:  # keep the API loud about missing artifacts
            raise RuntimeError(str(exc)) from exc

        model = bundle["model"]
        scaler = bundle["scaler"]
        selector = bundle["selector"]
        clinical_representation = bundle["clinical_representation"]
        dcqf = bundle["dcqf"]

        self.preprocessing.clinical_representation = clinical_representation
        self.preprocessing.set_fitted_state(selector=selector, scaler=scaler)
        self.preprocessing.dcqf = dcqf

        vector = self.preprocessing.transform_for_model(patient.frame_dict)
        prob = model.predict_proba(vector.reshape(1, -1))[0, 1]
        pred = int(prob >= 0.5)
        risk_percent = int(round(prob * 100))
        label = "Higher estimated risk" if pred == 1 else "Lower estimated risk"

        return PredictionResponse(
            prediction=pred,
            risk_probability=float(prob),
            risk_percentage=risk_percent,
            risk_label=label,
        )


prediction_service = PredictionService()
