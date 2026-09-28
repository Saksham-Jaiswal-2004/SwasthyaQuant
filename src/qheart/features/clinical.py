"""Clinical feature engineering for the 70,000-row cardiovascular dataset.

This module creates deterministic clinical representations.

No statistics are learned here and no target information is used.
Invalid physiological measurements are converted to NaN before
derived features are calculated. Missing-value handling remains
fold-local in the preprocessing pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["add_clinical_features"]


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically meaningful derived features."""

    out = df.copy()

    # ---------------------------------------------------------------
    # Validate source measurements before deriving secondary features
    # ---------------------------------------------------------------

    # Height and weight bounds.
    out.loc[
        ~out["height"].between(100, 250),
        "height",
    ] = np.nan

    out.loc[
        ~out["weight"].between(20, 300),
        "weight",
    ] = np.nan

    # Blood pressure component bounds.
    out.loc[
        ~out["ap_hi"].between(60, 250),
        "ap_hi",
    ] = np.nan

    out.loc[
        ~out["ap_lo"].between(30, 150),
        "ap_lo",
    ] = np.nan

    # A diastolic pressure cannot exceed systolic pressure.
    invalid_bp = (
        out["ap_hi"].notna()
        & out["ap_lo"].notna()
        & (out["ap_hi"] < out["ap_lo"])
    )

    out.loc[invalid_bp, ["ap_hi", "ap_lo"]] = np.nan

    # ---------------------------------------------------------------
    # Age
    # ---------------------------------------------------------------

    out["age_years"] = out["age"] / 365.25

    # ---------------------------------------------------------------
    # BMI
    # ---------------------------------------------------------------

    height_m = out["height"] / 100.0

    out["bmi"] = out["weight"] / (height_m ** 2)

    out.loc[
        ~out["bmi"].between(10, 80),
        "bmi",
    ] = np.nan

    # ---------------------------------------------------------------
    # Blood-pressure derived features
    # ---------------------------------------------------------------

    out["pulse_pressure"] = out["ap_hi"] - out["ap_lo"]

    out["map"] = (
        out["ap_lo"]
        + (out["ap_hi"] - out["ap_lo"]) / 3.0
    )

    # ---------------------------------------------------------------
    # Clinical categories
    # ---------------------------------------------------------------

    out["bmi_class"] = pd.cut(
        out["bmi"],
        bins=[-np.inf, 18.5, 25.0, 30.0, np.inf],
        labels=[
            "underweight",
            "normal",
            "overweight",
            "obese",
        ],
        right=False,
    )

    out["bp_group"] = pd.cut(
        out["ap_hi"],
        bins=[-np.inf, 120, 130, 140, 180, np.inf],
        labels=[
            "normal",
            "elevated",
            "stage1",
            "stage2",
            "crisis",
        ],
        right=False,
    )

    out["age_group"] = pd.cut(
        out["age_years"],
        bins=[-np.inf, 40, 50, 60, 70, np.inf],
        labels=[
            "young",
            "middle",
            "older",
            "senior",
            "elderly",
        ],
        right=False,
    )

    return out