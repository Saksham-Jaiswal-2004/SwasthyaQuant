from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from qheart import schema as S


class PatientInput(BaseModel):
    """Patient request body for the production inference API.

    The frontend uses human age in years, while the underlying qheart dataset stores
    age in days. The API accepts the human-readable value and converts it internally
    before calling the research pipeline so the exact repository feature logic is kept.
    """

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=18, le=120)
    height: int = Field(..., ge=100, le=250)
    weight: float = Field(..., ge=20.0, le=300.0)
    ap_hi: int = Field(..., ge=60, le=250)
    ap_lo: int = Field(..., ge=30, le=150)
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)
    gender: int = Field(..., ge=1, le=2)
    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)

    @field_validator("height", "weight", "ap_hi", "ap_lo")
    @classmethod
    def validate_plausible_ranges(cls, value: Any, info: Any) -> Any:
        name = info.field_name
        lo, hi = S.PLAUSIBLE[name]
        if value < lo or value > hi:
            raise ValueError(f"{name} must be within {S.PLAUSIBLE[name]}")
        return value

    @field_validator("ap_hi")
    @classmethod
    def validate_bp_relationship(cls, value: int, info: Any) -> int:
        if info.data.get("ap_lo") is not None and value < info.data["ap_lo"]:
            raise ValueError("ap_hi must be greater than or equal to ap_lo")
        return value

    @field_validator("ap_lo")
    @classmethod
    def validate_bp_relationship_lo(cls, value: int, info: Any) -> int:
        if info.data.get("ap_hi") is not None and info.data["ap_hi"] < value:
            raise ValueError("ap_lo must be less than or equal to ap_hi")
        return value

    @field_validator("gender", "cholesterol", "gluc")
    @classmethod
    def validate_categorical_values(cls, value: Any, info: Any) -> Any:
        name = info.field_name
        allowed = S.CATEGORIES[name]
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}")
        return value

    @property
    def frame_dict(self) -> dict[str, int | float]:
        age_for_pipeline = int(round(self.age * 365.25))
        return {
            "age": age_for_pipeline,
            "height": self.height,
            "weight": self.weight,
            "ap_hi": self.ap_hi,
            "ap_lo": self.ap_lo,
            "smoke": self.smoke,
            "alco": self.alco,
            "active": self.active,
            "gender": self.gender,
            "cholesterol": self.cholesterol,
            "gluc": self.gluc,
        }
