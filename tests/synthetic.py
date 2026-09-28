"""Synthetic stand-in for cardio_train.csv.

The synthetic dataset follows the 70k cardiovascular dataset schema so unit
tests do not depend on the real 70,000-row dataset.

The generator deliberately includes:
- missing continuous measurements,
- an impossible blood-pressure relationship,
- duplicate rows,
- and a feature-identical pair with disagreeing labels.

These traps allow data-quality and preprocessing tests to exercise the
failure modes the production pipeline must handle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


RAW_COLUMNS = [
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


def make_raw(
    n: int = 200,
    seed: int = 7,
    inject_traps: bool = True,
) -> pd.DataFrame:
    """Return a synthetic cardio_train.csv-style DataFrame."""

    if n < 20:
        raise ValueError("n must be at least 20")

    rng = np.random.default_rng(seed)

    age_years = np.clip(
        rng.normal(53, 7, n),
        30,
        65,
    )

    age = np.round(
        age_years * 365.25
    ).astype(int)

    gender = rng.choice(
        [1, 2],
        size=n,
        p=[0.35, 0.65],
    )

    height = np.clip(
        rng.normal(164, 8, n),
        145,
        195,
    ).round(1)

    weight = np.clip(
        rng.normal(75, 14, n),
        45,
        130,
    ).round(1)

    ap_hi = np.clip(
        rng.normal(130, 18, n),
        90,
        200,
    ).round().astype(int)

    ap_lo = np.clip(
        rng.normal(82, 10, n),
        50,
        120,
    ).round().astype(int)

    cholesterol = rng.choice(
        [1, 2, 3],
        size=n,
        p=[0.75, 0.18, 0.07],
    )

    gluc = rng.choice(
        [1, 2, 3],
        size=n,
        p=[0.85, 0.10, 0.05],
    )

    smoke = rng.choice(
        [0, 1],
        size=n,
        p=[0.9, 0.1],
    )

    alco = rng.choice(
        [0, 1],
        size=n,
        p=[0.93, 0.07],
    )

    active = rng.choice(
        [0, 1],
        size=n,
        p=[0.2, 0.8],
    )

    # A learnable but noisy cardiovascular-risk signal.
    bmi = weight / (height / 100.0) ** 2

    logit = (
        -4.0
        + 0.055 * (age_years - 50)
        + 0.018 * (ap_hi - 120)
        + 0.45 * (bmi - 25)
        + 0.55 * (cholesterol == 2)
        + 1.00 * (cholesterol == 3)
        + 0.35 * (gluc == 2)
        + 0.70 * (gluc == 3)
        + 0.45 * smoke
        - 0.20 * active
        + rng.normal(0, 0.7, n)
    )

    probability = 1.0 / (
        1.0 + np.exp(-logit)
    )

    cardio = (
        rng.random(n) < probability
    ).astype(int)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "age": age,
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active,
            "cardio": cardio,
        }
    )[RAW_COLUMNS]

    if inject_traps:
        # Missing source measurements.
        df.loc[0, "height"] = np.nan
        df.loc[1, "weight"] = np.nan

        # Missing blood-pressure measurements.
        df.loc[2, "ap_hi"] = np.nan
        df.loc[3, "ap_lo"] = np.nan

        # Impossible BP relationship.
        df.loc[4, "ap_hi"] = 90
        df.loc[4, "ap_lo"] = 110

        # Exact duplicate rows.
        df = pd.concat(
            [df, df.iloc[[5, 6]]],
            ignore_index=True,
        )

        # Feature-identical pair with disagreeing labels.
        twin = df.iloc[[10]].copy()
        twin["id"] = df["id"].max() + 1
        twin["cardio"] = 1 - int(
            twin["cardio"].iloc[0]
        )

        df = pd.concat(
            [df, twin],
            ignore_index=True,
        )

    return df


def write_raw(path, **kwargs) -> None:
    """Write a synthetic cardio_train.csv-style file."""

    make_raw(**kwargs).to_csv(
        path,
        sep=";",
        index=False,
    )