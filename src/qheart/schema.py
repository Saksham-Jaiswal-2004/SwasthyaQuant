from dataclasses import dataclass


TARGET = "cardio"

NUMERIC = [
    "age",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
]

BINARY_NUMERIC = [
    "smoke",
    "alco",
    "active",
]

CATEGORICAL = [
    "gender",
    "cholesterol",
    "gluc",
]

FEATURES = NUMERIC + BINARY_NUMERIC + CATEGORICAL

ALL_COLUMNS = FEATURES + [TARGET]


CATEGORIES = {
    "gender": [1, 2],
    "cholesterol": [1, 2, 3],
    "gluc": [1, 2, 3],
}


PLAUSIBLE = {
    "age": (3650, 36500),
    "height": (100, 250),
    "weight": (20, 300),
    "ap_hi": (60, 250),
    "ap_lo": (30, 150),
}


SENTINEL_ZERO = []

SUBGROUP_KEYS = [
    "gender",
    "age_band",
]


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str
    required: bool = True
    categories: list[int] | None = None


SPECS = {
    "age": ColumnSpec("age", "numeric"),
    "height": ColumnSpec("height", "numeric"),
    "weight": ColumnSpec("weight", "numeric"),
    "ap_hi": ColumnSpec("ap_hi", "numeric"),
    "ap_lo": ColumnSpec("ap_lo", "numeric"),

    "smoke": ColumnSpec("smoke", "binary"),
    "alco": ColumnSpec("alco", "binary"),
    "active": ColumnSpec("active", "binary"),

    "gender": ColumnSpec(
        "gender",
        "categorical",
        categories=CATEGORIES["gender"],
    ),
    "cholesterol": ColumnSpec(
        "cholesterol",
        "categorical",
        categories=CATEGORIES["cholesterol"],
    ),
    "gluc": ColumnSpec(
        "gluc",
        "categorical",
        categories=CATEGORIES["gluc"],
    ),

    "cardio": ColumnSpec("cardio", "target"),
}


def validate(
    df,
    *,
    require_target: bool = True,
) -> None:
    """Validate a dataframe against the 70k cardiovascular schema."""

    required = ALL_COLUMNS if require_target else FEATURES

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    for column, allowed in CATEGORIES.items():
        if column not in df.columns:
            continue

        observed = set(df[column].dropna().unique())
        invalid = observed - set(allowed)

        if invalid:
            raise ValueError(
                f"Column {column!r} contains unmapped categories: "
                f"{sorted(invalid)}; expected {allowed}"
            )

    if require_target:
        target = df[TARGET]

        if target.isna().any():
            raise ValueError(
                f"Target column {TARGET!r} contains NaN values"
            )

        observed_target = set(target.unique())

        if not observed_target.issubset({0, 1}):
            raise ValueError(
                f"Target column {TARGET!r} must be binary 0/1; "
                f"observed {sorted(observed_target)}"
            )