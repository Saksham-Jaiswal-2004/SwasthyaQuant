import numpy as np
import pandas as pd
import pytest

from qheart.features.select import (
    CANDIDATE_FEATURES,
    FoldSafeFeatureSelector,
)


def make_data(n=120, seed=123):
    rng = np.random.default_rng(seed)

    X = pd.DataFrame(
        {
            "age_years": rng.normal(53, 7, n),
            "bmi": rng.normal(27, 5, n),
            "ap_hi": rng.normal(127, 15, n),
            "ap_lo": rng.normal(81, 10, n),
            "pulse_pressure": rng.normal(46, 12, n),
            "map": rng.normal(96, 11, n),
            "cholesterol": rng.integers(1, 4, n),
            "gluc": rng.integers(1, 4, n),
            "active": rng.integers(0, 2, n),
            "gender": rng.integers(1, 3, n),
            "smoke": rng.integers(0, 2, n),
            "alco": rng.integers(0, 2, n),
        }
    )

    y = (X["ap_hi"] + 0.8 * X["age_years"] > 150).astype(int)

    return X, y


def test_selector_selects_requested_number_of_features():
    X, y = make_data()

    selector = FoldSafeFeatureSelector(k=8)
    selector.fit(X, y)

    assert len(selector.selected_features_) == 8
    assert len(selector.get_feature_names_out()) == 8
    assert selector.get_support().sum() == 8


def test_selector_transform_preserves_selected_order():
    X, y = make_data()

    selector = FoldSafeFeatureSelector(k=5)
    selector.fit(X, y)

    transformed = selector.transform(X)

    assert list(transformed.columns) == selector.selected_features_
    assert transformed.shape == (len(X), 5)


def test_selector_fails_before_fit():
    X, _ = make_data()
    selector = FoldSafeFeatureSelector(k=4)

    with pytest.raises(RuntimeError):
        selector.transform(X)


def test_selector_rejects_missing_candidate():
    X, y = make_data()
    X = X.drop(columns=["bmi"])

    selector = FoldSafeFeatureSelector(k=4)

    with pytest.raises(ValueError, match="missing candidate features"):
        selector.fit(X, y)


def test_selector_handles_missing_values():
    X, y = make_data()

    X.loc[0, "bmi"] = np.nan
    X.loc[1, "ap_hi"] = np.nan

    selector = FoldSafeFeatureSelector(k=4)
    selector.fit(X, y)

    transformed = selector.transform(X)

    assert transformed.shape == (len(X), 4)
    assert selector.selected_features_


def test_selector_is_training_fold_local():
    X, y = make_data(n=100)

    train = X.iloc[:80].copy()
    y_train = y[:80]

    selector_a = FoldSafeFeatureSelector(k=6)
    selector_a.fit(train, y_train)

    # Completely unrelated extra rows are not part of the training fold.
    extra = X.iloc[80:].copy()
    extra.loc[:, "ap_hi"] = 10000
    extra.loc[:, "bmi"] = 1000

    selector_b = FoldSafeFeatureSelector(k=6)
    selector_b.fit(train, y_train)

    assert selector_a.selected_features_ == selector_b.selected_features_
    np.testing.assert_allclose(
        selector_a.mi_.values,
        selector_b.mi_.values,
    )


def test_candidate_features_are_unique():
    assert len(CANDIDATE_FEATURES) == len(set(CANDIDATE_FEATURES))