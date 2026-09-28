"""Leakage-safe preprocessing and clinical representation.

The project has two preprocessing paths:

1. Raw-schema preprocessing
   Used by existing classical/raw-feature components and tests.

2. Clinical compact representation
   Used by the 70k cardiovascular modelling pipeline.

Any learned operation is fitted only when the enclosing model is fitted.
When that model is used inside cross-validation, this makes the operation
fold-local automatically.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from qheart import schema as S
from qheart.features.clinical import add_clinical_features
from qheart.features.select import (
    CANDIDATE_FEATURES,
    FoldSafeFeatureSelector,
)


# ---------------------------------------------------------------------------
# Raw-schema feature names
# ---------------------------------------------------------------------------

def feature_names(
    *,
    add_missing_indicators: bool = True,
) -> list[str]:
    """Return deterministic encoded feature names.

    Categorical variables use drop-first encoding. This keeps the encoded
    design matrix full rank for linear models.
    """

    names = list(S.NUMERIC)
    names += list(S.BINARY_NUMERIC)

    # Drop the first category and retain the remaining categories.
    for column in S.CATEGORICAL:
        categories = S.CATEGORIES.get(column, [])

        for category in categories[1:]:
            names.append(
                f"{column}={category}"
            )

    if add_missing_indicators:
        names.extend(
            f"{column}__missing"
            for column in S.NUMERIC
        )

    return names


# ---------------------------------------------------------------------------
# Raw-schema deterministic encoding
# ---------------------------------------------------------------------------

def encode_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Encode the raw cardiovascular dataframe.

    No statistics are learned here.

    Categorical variables use drop-first encoding:
        gender:       one column
        cholesterol:  two columns
        gluc:         two columns

    This function is deliberately compatible with the original
    preprocessing contract and its tests.
    """

    S.validate(df)

    out = pd.DataFrame(
        index=df.index
    )

    # Continuous numeric variables.
    for column in S.NUMERIC:
        out[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Binary numeric variables.
    for column in S.BINARY_NUMERIC:
        out[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Drop-first categorical encoding.
    for column in S.CATEGORICAL:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        categories = S.CATEGORIES[column]

        for category in categories[1:]:
            out[f"{column}={category}"] = (
                values == category
            ).astype(float)

    # Missing indicators.
    if S.SENTINEL_ZERO:
        for column in S.SENTINEL_ZERO:
            if column in out.columns:
                out[f"{column}__missing"] = (
                    pd.to_numeric(
                        df[column],
                        errors="coerce",
                    ).isna().astype(float)
                )
    else:
        for column in S.NUMERIC:
            out[f"{column}__missing"] = (
                pd.to_numeric(
                    df[column],
                    errors="coerce",
                ).isna().astype(float)
            )

    return out


# ---------------------------------------------------------------------------
# Raw-schema preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor(
    *,
    scale: bool = True,
    add_missing_indicators: bool = True,
) -> ColumnTransformer:
    """Build the unfitted raw-schema preprocessor.

    The returned transformer can operate on the encoded DataFrame produced
    by ``encode_frame``. It also supports NumPy matrices for compatibility
    with the existing preprocessing tests.
    """

    names = feature_names(
        add_missing_indicators=add_missing_indicators
    )

    numeric = list(S.NUMERIC)

    encoded = [
        name
        for name in names
        if name not in numeric
    ]

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]

    if scale:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler()
            )
        )

    numeric_pipeline = Pipeline(
        numeric_steps
    )

    encoded_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            )
        ]
    )

    # Integer column positions are important here.
    #
    # Using positional indices means this transformer works both with
    # DataFrames and NumPy arrays, which preserves the existing API.
    numeric_idx = [
        names.index(column)
        for column in numeric
    ]

    encoded_idx = [
        names.index(column)
        for column in encoded
    ]

    return ColumnTransformer(
        [
            (
                "numeric",
                numeric_pipeline,
                numeric_idx,
            ),
            (
                "encoded",
                encoded_pipeline,
                encoded_idx,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Fold-safe clinical representation
# ---------------------------------------------------------------------------

class ClinicalRepresentation(
    BaseEstimator,
    TransformerMixin,
):
    """Fold-safe compact clinical representation.

    The transformation sequence is:

        raw DataFrame
            -> deterministic clinical features
            -> MI/mRMR selection
            -> median imputation
            -> standardisation

    Feature selection and imputation statistics are learned only during
    ``fit``. When this transformer is a Pipeline step, that means they are
    learned from the training fold only.
    """

    def __init__(
        self,
        k: int = 8,
        redundancy_penalty: float = 0.5,
        output: str = "standard",
        random_state: int = 20260830,
    ):
        self.k = k
        self.redundancy_penalty = redundancy_penalty
        self.output = output
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
    ):
        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "ClinicalRepresentation.fit expects "
                "a pandas DataFrame."
            )

        if self.output not in {
            "standard",
            "angles",
        }:
            raise ValueError(
                "output must be 'standard' or 'angles'."
            )

        clinical = add_clinical_features(
            X.copy()
        )

        missing = [
            column
            for column in CANDIDATE_FEATURES
            if column not in clinical.columns
        ]

        if missing:
            raise ValueError(
                "Missing clinical candidate features: "
                f"{missing}"
            )

        candidate = clinical.loc[
            :,
            CANDIDATE_FEATURES,
        ]

        # Fit selector ONLY on training data.
        self.selector_ = FoldSafeFeatureSelector(
            k=self.k,
            redundancy_penalty=self.redundancy_penalty,
            random_state=self.random_state,
        )

        selected = self.selector_.fit_transform(
            candidate,
            y,
        )

        self.selected_features_ = list(
            self.selector_.get_feature_names_out()
        )

        # Fit imputation ONLY on training data.
        self.imputer_ = SimpleImputer(
            strategy="median"
        )

        imputed = self.imputer_.fit_transform(
            selected
        )

        if self.output == "standard":
            self.scaler_ = StandardScaler()

            self.scaler_.fit(
                imputed
            )

        else:
            self.scaler_ = FixedAngleScaler(
                self.selected_features_
            )

            self.scaler_.fit(
                imputed
            )

        self.n_features_in_ = len(
            CANDIDATE_FEATURES
        )

        self.n_features_out_ = len(
            self.selected_features_
        )

        return self

    def transform(
        self,
        X,
    ):
        if not hasattr(
            self,
            "selector_",
        ):
            raise RuntimeError(
                "ClinicalRepresentation must be fitted "
                "before transform()."
            )

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "ClinicalRepresentation.transform expects "
                "a pandas DataFrame."
            )

        clinical = add_clinical_features(
            X.copy()
        )

        candidate = clinical.loc[
            :,
            CANDIDATE_FEATURES,
        ]

        selected = self.selector_.transform(
            candidate
        )

        imputed = self.imputer_.transform(
            selected
        )

        transformed = self.scaler_.transform(
            imputed
        )

        transformed = np.asarray(
            transformed,
            dtype=float,
        )

        if not np.isfinite(
            transformed
        ).all():
            raise ValueError(
                "ClinicalRepresentation produced "
                "non-finite values."
            )

        return transformed

    def get_feature_names_out(
        self,
        input_features=None,
    ):
        if not hasattr(
            self,
            "selected_features_",
        ):
            raise RuntimeError(
                "ClinicalRepresentation has not been fitted."
            )

        return np.asarray(
            self.selected_features_,
            dtype=object,
        )


def build_clinical_representation(
    *,
    k: int = 8,
    redundancy_penalty: float = 0.5,
    output: str = "standard",
    random_state: int = 20260830,
) -> ClinicalRepresentation:
    """Return an unfitted fold-safe clinical representation."""

    return ClinicalRepresentation(
        k=k,
        redundancy_penalty=redundancy_penalty,
        output=output,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Fixed-angle scaling
# ---------------------------------------------------------------------------

# The existing tests and raw-schema quantum path expect schema plausibility
# bounds. Clinical derived features have explicit fixed bounds below.

CLINICAL_ANGLE_BOUNDS = {
    "age_years": (18.0, 100.0),
    "bmi": (10.0, 80.0),
    "ap_hi": (60.0, 250.0),
    "ap_lo": (30.0, 150.0),
    "pulse_pressure": (0.0, 220.0),
    "map": (30.0, 200.0),
    "cholesterol": (1.0, 3.0),
    "gluc": (1.0, 3.0),
    "active": (0.0, 1.0),
    "gender": (1.0, 2.0),
    "smoke": (0.0, 1.0),
    "alco": (0.0, 1.0),
}


class FixedAngleScaler(
    BaseEstimator,
    TransformerMixin,
):
    """Map features to [0, pi] using fixed, non-learned bounds.

    Raw-schema features use ``schema.PLAUSIBLE``.

    Clinical derived features use ``CLINICAL_ANGLE_BOUNDS``.

    No statistics are learned from the dataset.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
    ):
        self.feature_names = list(
            feature_names
        )

    def _bounds(
        self,
        name: str,
    ) -> tuple[float, float]:
        if name in CLINICAL_ANGLE_BOUNDS:
            return CLINICAL_ANGLE_BOUNDS[name]

        if name in S.PLAUSIBLE:
            return S.PLAUSIBLE[name]

        # Fixed 0-1 bounds for binary / missing indicators.
        if (
            name in S.BINARY_NUMERIC
            or name.endswith("__missing")
        ):
            return 0.0, 1.0

        # One-hot columns are also fixed 0/1.
        if "=" in name:
            return 0.0, 1.0

        raise ValueError(
            f"No fixed angle bounds defined for '{name}'."
        )

    def fit(
        self,
        X,
        y=None,
    ):
        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        if X.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                "expected "
                f"{len(self.feature_names)} columns, "
                f"got {X.shape[1]}."
            )

        # Resolve all bounds during fit so that transform is deterministic.
        self.bounds_ = [
            self._bounds(name)
            for name in self.feature_names
        ]

        return self

    def transform(
        self,
        X,
    ):
        X = np.asarray(
            X,
            dtype=float,
        )

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2D array."
            )

        if X.shape[1] != len(
            self.feature_names
        ):
            raise ValueError(
                "expected "
                f"{len(self.feature_names)} columns, "
                f"got {X.shape[1]}."
            )

        # ``fit`` is optional for backwards compatibility because the
        # scaler itself learns nothing from the data. Resolve bounds here
        # when transform() is called directly.
        bounds = getattr(
            self,
            "bounds_",
            [
                self._bounds(name)
                for name in self.feature_names
            ],
        )

        out = np.empty_like(
            X,
            dtype=float,
        )

        for j, (
            lo,
            hi,
        ) in enumerate(bounds):
            values = X[:, j]

            # Fixed scaling is only meaningful for finite values.
            if not np.isfinite(
                values
            ).all():
                raise ValueError(
                    "FixedAngleScaler received "
                    f"non-finite values in '{self.feature_names[j]}'."
                )

            scaled = (
                (values - lo)
                / (hi - lo)
            )

            scaled = np.clip(
                scaled,
                0.0,
                1.0,
            )

            out[:, j] = (
                scaled * np.pi
            )

        return out