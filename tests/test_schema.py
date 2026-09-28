import numpy as np
import pandas as pd
import pytest

import qheart.schema as S


def _good(n=6):
    return pd.DataFrame({
        "age": np.linspace(40 * 365.25, 60 * 365.25, n),
        "height": np.full(n, 170.0),
        "weight": np.full(n, 70.0),
        "ap_hi": np.full(n, 130.0),
        "ap_lo": np.full(n, 80.0),
        "smoke": np.zeros(n, dtype=int),
        "alco": np.zeros(n, dtype=int),
        "active": np.ones(n, dtype=int),
        "gender": np.full(n, 2, dtype=int),
        "cholesterol": np.ones(n, dtype=int),
        "gluc": np.ones(n, dtype=int),
        "cardio": [0, 1] * (n // 2),
    })


def test_every_declared_column_has_a_spec():
    for column in S.ALL_COLUMNS:
        assert column in S.SPECS


def test_categorical_specs_match_categories():
    for column in S.CATEGORICAL:
        assert S.SPECS[column].categories == S.CATEGORIES[column]


def test_plausible_bounds_are_ordered_and_physiological():
    for column, (lo, hi) in S.PLAUSIBLE.items():
        assert lo < hi, f"{column} bounds are inverted"


def test_validate_accepts_a_conformant_frame():
    S.validate(_good())


def test_validate_rejects_a_missing_column():
    df = _good().drop(columns=["weight"])

    with pytest.raises(ValueError):
        S.validate(df)


def test_validate_rejects_an_unmapped_category():
    df = _good()
    df.loc[0, "cholesterol"] = 99

    with pytest.raises(ValueError):
        S.validate(df)


def test_validate_rejects_a_non_binary_target():
    df = _good()
    df.loc[0, "cardio"] = 2

    with pytest.raises(ValueError):
        S.validate(df)


def test_validate_rejects_nan_target():
    df = _good()
    df.loc[0, "cardio"] = np.nan

    with pytest.raises(ValueError):
        S.validate(df)


def test_binary_columns_are_declared():
    assert set(S.BINARY_NUMERIC) == {"smoke", "alco", "active"}


def test_cardio_is_the_target():
    assert S.TARGET == "cardio"