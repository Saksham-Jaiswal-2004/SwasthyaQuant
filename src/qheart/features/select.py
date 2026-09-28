"""Fold-safe feature selection for the 70k cardiovascular dataset.

The selector is deliberately fitted only on the training data supplied to ``fit``.
It ranks candidate features by mutual information with the target while penalising
redundancy with features already selected.

This is an mRMR-style selector rather than a plain SelectKBest implementation:
highly correlated representations of the same physiological signal should not
consume the entire compact feature budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "CANDIDATE_FEATURES",
    "FoldSafeFeatureSelector",
    "select_columns",
    "describe",
]


CANDIDATE_FEATURES = [
    "age_years",
    "bmi",
    "ap_hi",
    "ap_lo",
    "pulse_pressure",
    "map",
    "cholesterol",
    "gluc",
    "active",
    "gender",
    "smoke",
    "alco",
]


RATIONALE = {
    "age_years": "Continuous age representation; captures age-related cardiovascular risk.",
    "bmi": "Body-mass index; captures weight relative to height.",
    "ap_hi": "Systolic blood pressure; strong cardiovascular risk signal.",
    "ap_lo": "Diastolic blood pressure; complementary blood-pressure information.",
    "pulse_pressure": "Systolic minus diastolic pressure; captures pressure amplitude.",
    "map": "Mean arterial pressure; compact representation of overall arterial pressure.",
    "cholesterol": "Ordinal cholesterol category with substantial target-rate separation.",
    "gluc": "Ordinal glucose category with measurable target-rate separation.",
    "active": "Physical-activity indicator representing a behavioral dimension.",
    "gender": "Demographic variable retained as an initial candidate for subgroup analysis.",
    "smoke": "Smoking indicator; included as a candidate despite weak marginal association.",
    "alco": "Alcohol-use indicator; included as a candidate despite weak marginal association.",
}


def _validate_input(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("FoldSafeFeatureSelector expects a pandas DataFrame")

    missing = [c for c in CANDIDATE_FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"missing candidate features: {missing}")


class FoldSafeFeatureSelector:
    """Training-fold-only mRMR-style selector.

    Parameters
    ----------
    k:
        Number of features to select.
    redundancy_penalty:
        Weight applied to the maximum absolute correlation with already-selected
        features. Zero reduces the method to greedy mutual-information ranking.
    random_state:
        Seed passed to ``mutual_info_classif``.

    Notes
    -----
    All fitted quantities are created in ``fit``. ``transform`` only applies the
    resulting feature list and therefore cannot learn from validation/test rows.
    """

    def __init__(
        self,
        k: int = 8,
        redundancy_penalty: float = 0.5,
        random_state: int = 20260830,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")
        if redundancy_penalty < 0:
            raise ValueError("redundancy_penalty must be >= 0")

        self.k = int(k)
        self.redundancy_penalty = float(redundancy_penalty)
        self.random_state = int(random_state)

        self.selected_features_: list[str] | None = None
        self.mi_: pd.Series | None = None
        self.selection_scores_: dict[str, float] | None = None

    def fit(self, X: pd.DataFrame, y) -> "FoldSafeFeatureSelector":
        _validate_input(X)

        y = np.asarray(y).ravel()
        if len(X) != len(y):
            raise ValueError("X and y have different numbers of rows")

        if len(X) == 0:
            raise ValueError("cannot fit selector on an empty dataset")

        from sklearn.feature_selection import mutual_info_classif

        data = X[CANDIDATE_FEATURES].copy()

        # Median imputation is fitted on THIS training fold only.
        medians = data.median(numeric_only=True)
        data = data.fillna(medians)

        # All candidate features are numeric/ordinal at this stage.
        values = data.to_numpy(dtype=float)

        mi = mutual_info_classif(
            values,
            y,
            random_state=self.random_state,
        )

        self.mi_ = pd.Series(
            mi,
            index=CANDIDATE_FEATURES,
            dtype=float,
        ).sort_values(ascending=False)

        # Feature-feature redundancy is also calculated only from this training fold.
        corr = data.corr(method="spearman").abs().fillna(0.0)

        n_select = min(self.k, len(CANDIDATE_FEATURES))
        selected: list[str] = []
        scores: dict[str, float] = {}

        remaining = list(CANDIDATE_FEATURES)

        # First feature: highest training-fold MI.
        first = self.mi_.idxmax()
        selected.append(first)
        remaining.remove(first)
        scores[first] = float(self.mi_[first])

        while remaining and len(selected) < n_select:
            best_feature = None
            best_score = -np.inf

            for feature in remaining:
                redundancy = max(
                    float(corr.loc[feature, chosen])
                    for chosen in selected
                )

                score = float(
                    self.mi_[feature]
                    - self.redundancy_penalty * redundancy
                )

                # Deterministic alphabetical tie-breaking.
                if (
                    score > best_score
                    or (
                        np.isclose(score, best_score)
                        and (best_feature is None or feature < best_feature)
                    )
                ):
                    best_score = score
                    best_feature = feature

            assert best_feature is not None
            selected.append(best_feature)
            remaining.remove(best_feature)
            scores[best_feature] = float(best_score)

        self.selected_features_ = selected
        self.selection_scores_ = scores

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise RuntimeError("call fit before transform")

        _validate_input(X)
        return X.loc[:, self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_support(self) -> np.ndarray:
        if self.selected_features_ is None:
            raise RuntimeError("call fit before get_support")

        return np.asarray(
            [feature in self.selected_features_ for feature in CANDIDATE_FEATURES],
            dtype=bool,
        )

    def get_feature_names_out(self) -> np.ndarray:
        if self.selected_features_ is None:
            raise RuntimeError("call fit before get_feature_names_out")

        return np.asarray(self.selected_features_, dtype=object)


def select_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Select named columns while preserving their order."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return df.loc[:, columns].copy()


def describe(columns: list[str] | None = None) -> pd.DataFrame:
    """Return descriptions for the candidate or selected feature list."""
    cols = CANDIDATE_FEATURES if columns is None else list(columns)

    missing = [c for c in cols if c not in RATIONALE]
    if missing:
        raise KeyError(f"unknown features: {missing}")

    return pd.DataFrame(
        {
            "feature": cols,
            "rationale": [RATIONALE[c] for c in cols],
        }
    )