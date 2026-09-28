import numpy as np
import pandas as pd

from qheart.features.clinical import add_clinical_features


def test_clinical_features_are_created():
    df = pd.DataFrame(
        {
            "age": [18393],
            "height": [168],
            "weight": [62.0],
            "ap_hi": [110],
            "ap_lo": [80],
            "smoke": [0],
            "alco": [0],
            "active": [1],
            "gender": [2],
            "cholesterol": [1],
            "gluc": [1],
            "cardio": [0],
        }
    )

    out = add_clinical_features(df)

    assert "age_years" in out
    assert "bmi" in out
    assert "pulse_pressure" in out
    assert "map" in out
    assert "bmi_class" in out
    assert "bp_group" in out
    assert "age_group" in out

    assert np.isclose(out.loc[0, "bmi"], 21.9671, atol=1e-3)
    assert np.isclose(out.loc[0, "pulse_pressure"], 30.0)
    assert np.isclose(out.loc[0, "map"], 90.0)


def test_invalid_blood_pressure_becomes_missing():
    df = pd.DataFrame(
        {
            "age": [18393],
            "height": [168],
            "weight": [62.0],
            "ap_hi": [90],
            "ap_lo": [110],
            "smoke": [0],
            "alco": [0],
            "active": [1],
            "gender": [2],
            "cholesterol": [1],
            "gluc": [1],
            "cardio": [0],
        }
    )

    out = add_clinical_features(df)

    assert pd.isna(out.loc[0, "ap_hi"])
    assert pd.isna(out.loc[0, "ap_lo"])
    assert pd.isna(out.loc[0, "pulse_pressure"])
    assert pd.isna(out.loc[0, "map"])


def test_implausible_height_and_weight_become_missing():
    df = pd.DataFrame(
        {
            "age": [18393],
            "height": [75],
            "weight": [10.0],
            "ap_hi": [120],
            "ap_lo": [80],
            "smoke": [0],
            "alco": [0],
            "active": [1],
            "gender": [2],
            "cholesterol": [1],
            "gluc": [1],
            "cardio": [0],
        }
    )

    out = add_clinical_features(df)

    assert pd.isna(out.loc[0, "height"])
    assert pd.isna(out.loc[0, "weight"])
    assert pd.isna(out.loc[0, "bmi"])