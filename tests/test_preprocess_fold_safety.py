"""Fold-safety tests for the 70k cardiovascular dataset.

Leakage can make metrics look better while remaining invisible to ordinary
model-performance tests, so preprocessing invariants are tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qheart import schema as S
from qheart.preprocess.pipeline import (
    FixedAngleScaler,
    encode_frame,
    feature_names,
)


def test_encoding_layout_is_fixed_by_schema_not_by_data(clean_df):
    """Encoding must have the same columns regardless of observed categories."""
    full = encode_frame(clean_df)

    # Use a subset containing only one observed level for each categorical
    # variable. The encoded layout must still be determined by the schema.
    subset = clean_df.iloc[:100].copy()

    encoded_subset = encode_frame(subset)

    assert list(full.columns) == list(encoded_subset.columns)
    assert list(full.columns) == feature_names()


def test_encoding_is_stateless_so_it_cannot_leak(clean_df):
    """Encoding rows independently must give the same representation."""
    a = encode_frame(clean_df.iloc[:100])
    b = encode_frame(clean_df.iloc[:100].copy())
    c = encode_frame(clean_df).iloc[:100]

    assert np.allclose(
        a.to_numpy(),
        b.to_numpy(),
        equal_nan=True,
    )

    assert np.allclose(
        a.to_numpy(),
        c.to_numpy(),
        equal_nan=True,
    )


def test_drop_first_keeps_the_design_matrix_full_rank(clean_df):
    """Categorical encoding must drop exactly one reference level."""
    X = encode_frame(clean_df)

    for column in S.CATEGORICAL:
        cols = [
            c for c in X.columns
            if c.startswith(f"{column}=")
        ]

        assert len(cols) == len(S.CATEGORIES[column]) - 1, (
            f"{column} must drop one level"
        )


def test_encode_frame_preserves_missing_values_before_imputation(clean_df):
    """Encoding must not silently fill missing source measurements."""
    df = clean_df.copy()

    # Introduce missingness into a continuous source feature.
    df.loc[df.index[0], "height"] = np.nan

    X = encode_frame(df)

    assert np.isnan(X.loc[df.index[0], "height"])


def test_encode_frame_validates_its_input(clean_df):
    """Out-of-vocabulary categorical values must be rejected."""
    bad = clean_df.copy()

    # Convert to object first so pandas permits construction of the invalid
    # value. The schema validator must still reject it.
    bad["gender"] = bad["gender"].astype(object)
    bad.loc[bad.index[0], "gender"] = 99

    with pytest.raises(ValueError):
        encode_frame(bad)


def test_fixed_angle_scaler_learns_nothing():
    """FixedAngleScaler must be independent of the data passed to fit()."""
    names = feature_names()

    rng = np.random.default_rng(0)
    X = rng.uniform(
        20,
        200,
        size=(50, len(names)),
    )

    s1 = FixedAngleScaler(names)
    s2 = FixedAngleScaler(names)

    s1.fit(X[:5])
    s2.fit(X)

    assert np.allclose(
        s1.transform(X),
        s2.transform(X),
    )


def test_fixed_angle_scaler_output_is_inside_the_safe_band():
    """Angle encoding must remain inside [0, pi]."""
    names = feature_names()

    rng = np.random.default_rng(1)

    # Deliberately extreme inputs.
    X = rng.uniform(
        -1e4,
        1e4,
        size=(40, len(names)),
    )

    Z = FixedAngleScaler(names).transform(X)

    assert Z.min() >= 0.0
    assert Z.max() <= np.pi + 1e-12


def test_fixed_angle_scaler_uses_clinical_bounds_for_named_features():
    """Named clinical features must use schema plausibility bounds."""
    names = ["age"]

    lo, hi = S.PLAUSIBLE["age"]

    scaler = FixedAngleScaler(names)

    Z = scaler.transform(
        np.array([
            [lo],
            [hi],
        ])
    )

    assert np.isclose(Z[0, 0], 0.0)
    assert np.isclose(Z[1, 0], np.pi)


def test_fixed_angle_scaler_rejects_a_width_mismatch():
    """The scaler must reject matrices with the wrong feature width."""
    scaler = FixedAngleScaler(feature_names())

    with pytest.raises(ValueError, match="expected"):
        scaler.transform(
            np.zeros((3, 2))
        )


@pytest.mark.needs_sklearn
def test_preprocessor_is_returned_unfitted():
    """The preprocessor must be fitted by the CV fold, not at construction."""
    from sklearn.exceptions import NotFittedError

    from qheart.preprocess import build_preprocessor

    prep = build_preprocessor()

    with pytest.raises(NotFittedError):
        prep.transform(
            np.zeros(
                (3, len(feature_names()))
            )
        )


@pytest.mark.needs_sklearn
def test_imputation_statistics_come_only_from_the_training_rows():
    """Imputation statistics must be learned exclusively from training data."""
    from qheart.preprocess import build_preprocessor

    names = feature_names()
    n = 200

    X = np.zeros(
        (n, len(names))
    )

    age_index = names.index("age")

    # Training rows contain age=50.
    # Test rows contain age=90.
    X[:, age_index] = np.r_[
        np.full(100, 50.0),
        np.full(100, 90.0),
    ]

    # Missing value in the training distribution.
    X[0, age_index] = np.nan

    train = np.arange(1, 100)

    prep = build_preprocessor(scale=False)
    prep.fit(X[train])

    filled = prep.transform(
        X[[0]]
    )[0, 0]

    assert np.isclose(
        filled,
        50.0,
    ), (
        "imputed value must come from the training fold's median, "
        "not from the whole dataset"
    )


@pytest.mark.needs_sklearn
def test_model_pipelines_carry_their_preprocessor_as_a_step():
    """Every classical model must fit preprocessing inside the CV pipeline."""
    from qheart.models.classical import CLASSICAL

    for name, factory in CLASSICAL.items():
        pipe = factory()

        assert "prep" in dict(pipe.steps), (
            f"{name} must fit preprocessing inside the fold"
        )

        assert list(dict(pipe.steps)) == [
            "prep",
            "model",
        ]