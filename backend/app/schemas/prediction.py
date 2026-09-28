from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Model-estimated risk response.

    The probability is produced by the trained model's predict_proba[:, 1] path and
    is intentionally described as risk_estimate without implying a diagnosis.
    """

    prediction: int = Field(..., ge=0, le=1)
    risk_probability: float = Field(..., ge=0.0, le=1.0)
    risk_percentage: int = Field(..., ge=0, le=100)
    risk_label: str = Field(...)
    disclaimer: str = (
        "Model-estimated risk only. This is not a medical diagnosis or clinical advice."
    )
