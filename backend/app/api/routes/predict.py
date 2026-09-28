from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.patient import PatientInput
from app.schemas.prediction import PredictionResponse
from app.services.prediction_service import prediction_service

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientInput) -> PredictionResponse:
    try:
        return prediction_service.predict(patient)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
