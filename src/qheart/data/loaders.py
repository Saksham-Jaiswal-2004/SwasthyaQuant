"""Source adapter for the 70,000-row cardiovascular disease dataset.

The loader is responsible only for:
    1. Reading the raw dataset.
    2. Validating the expected source columns.
    3. Normalising the target name.
    4. Validating categorical domains and the binary target.

It does NOT:
    - impute missing values
    - scale features
    - engineer features
    - filter rows
    - clip outliers

Those operations belong to the preprocessing/feature-engineering pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from qheart import schema as S

__all__ = ["load", "register_loader", "available"]


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., pd.DataFrame]] = {}


def register_loader(name: str):
    """Register a dataset loader under a source name."""

    def deco(fn):
        _REGISTRY[name] = fn
        return fn

    return deco


def available() -> list[str]:
    """Return all registered dataset sources."""

    return sorted(_REGISTRY)


def load(
    source: str,
    path: str | Path,
    **kw,
) -> pd.DataFrame:
    """Load a registered dataset and validate its canonical schema."""

    if source not in _REGISTRY:
        raise KeyError(
            f"unknown source {source!r}; available: {available()}"
        )

    df = _REGISTRY[source](Path(path), **kw)

    return _finalise(df, source)


# ---------------------------------------------------------------------------
# 70,000-row cardiovascular disease dataset
# ---------------------------------------------------------------------------

_CARDIO_70000_COLUMNS = [
    "id",
    "age",
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio",
]


@register_loader("cardio_70000")
def _load_cardio_70000(path: Path) -> pd.DataFrame:
    """Load the 70,000-row cardiovascular disease dataset.

    Expected raw columns:

        id
        age
        gender
        height
        weight
        ap_hi
        ap_lo
        cholesterol
        gluc
        smoke
        alco
        active
        cardio

    The raw CSV uses semicolon delimiters.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found."
        )

    # The original cardiovascular dataset uses ';' as delimiter.
    df = pd.read_csv(path, sep=";")

    # Validate the source structure before doing anything else.
    missing = sorted(
        set(_CARDIO_70000_COLUMNS) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name} is missing expected columns {missing}. "
            f"Found: {sorted(df.columns)}."
        )


    

    # Keep only the expected source columns.
    df = df[_CARDIO_70000_COLUMNS].copy()

    df = df.drop(columns=["id"])

    # Rename source target to the canonical qheart target.
    df = df.rename(
        columns={"cardio": S.TARGET}
    )

    # Explicit numeric conversion.
    numeric_columns = [
        "age",
        "gender",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        S.TARGET,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ---------------------------------------------------------------------------
# Final canonical-schema validation
# ---------------------------------------------------------------------------

def _finalise(
    df: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """Validate and finalise the canonical cardiovascular dataset."""

    # Required columns.
    missing = sorted(
        set(S.ALL_COLUMNS) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"loader {source!r} did not produce required "
            f"columns {missing}"
        )

    # Keep canonical ordering.
    df = df[S.ALL_COLUMNS].copy()

    # -----------------------------------------------------------------------
    # Validate categorical variables
    # -----------------------------------------------------------------------

    for column, allowed in S.CATEGORIES.items():

        if column not in df.columns:
            continue

        seen = set(
            df[column]
            .dropna()
            .unique()
        )

        illegal = sorted(
            seen - set(allowed)
        )

        if illegal:
            raise ValueError(
                f"{column!r} contains illegal values {illegal}; "
                f"allowed values are {list(allowed)}."
            )

        df[column] = pd.Categorical(
            df[column],
            categories=allowed,
        )

    # -----------------------------------------------------------------------
    # Validate target
    # -----------------------------------------------------------------------

    df[S.TARGET] = pd.to_numeric(
        df[S.TARGET],
        errors="raise",
    ).astype(int)

    bad_target = sorted(
        set(df[S.TARGET].unique()) - {0, 1}
    )

    if bad_target:
        raise ValueError(
            f"target must contain only 0/1, "
            f"found {bad_target}"
        )

    # -----------------------------------------------------------------------
    # Dataset metadata
    # -----------------------------------------------------------------------

    df.attrs["source"] = source

    return df.reset_index(drop=True)