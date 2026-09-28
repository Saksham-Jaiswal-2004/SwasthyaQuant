from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from qheart.data.loaders import load
from qheart import schema as S


SEED = 20260830
PATH = "data/raw/cardio_train.csv"


def make_group_ids(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """
    Create deterministic group IDs without pandas categorical
    Cartesian-product expansion.
    """
    return pd.util.hash_pandas_object(
        df.loc[:, columns],
        index=False,
    )


def main() -> None:
    print("=" * 72)
    print("STAGE B.1 — DUPLICATE GROUP AUDIT")
    print("=" * 72)

    df = load("cardio_70000", PATH)
    features = list(S.FEATURES)

    print(f"Dataset shape: {df.shape}")
    print(f"Feature columns: {len(features)}")

    # ==================================================================
    # 1. Exact duplicates: features + target
    # ==================================================================

    exact_cols = features + [S.TARGET]
    exact_group = make_group_ids(df, exact_cols)

    exact_counts = exact_group.value_counts()
    exact_duplicate_groups = exact_counts[exact_counts > 1]

    exact_duplicate_rows = int(
        exact_counts[exact_counts > 1].sum()
    )

    print("\nExact duplicate groups")
    print("-" * 72)
    print(
        f"Rows belonging to exact duplicate groups: "
        f"{exact_duplicate_rows}"
    )
    print(
        f"Number of exact duplicate groups: "
        f"{len(exact_duplicate_groups)}"
    )

    if len(exact_duplicate_groups):
        print("\nLargest exact duplicate group sizes:")
        print(exact_duplicate_groups.head(10).to_string())

    # ==================================================================
    # 2. Feature-identical groups, ignoring target
    # ==================================================================

    feature_group = make_group_ids(df, features)

    feature_counts = feature_group.value_counts()
    feature_duplicate_groups = feature_counts[
        feature_counts > 1
    ]

    feature_duplicate_rows = int(
        feature_counts[feature_counts > 1].sum()
    )

    print("\nFeature-identical groups")
    print("-" * 72)
    print(
        f"Rows belonging to feature-duplicate groups: "
        f"{feature_duplicate_rows}"
    )
    print(
        f"Number of feature-duplicate groups: "
        f"{len(feature_duplicate_groups)}"
    )

    # ==================================================================
    # 3. Identify conflicting-label duplicate groups
    # ==================================================================

    duplicate_mask = feature_group.isin(
        feature_duplicate_groups.index
    )

    duplicate_df = pd.DataFrame(
        {
            "group": feature_group[duplicate_mask].to_numpy(),
            "target": df.loc[duplicate_mask, S.TARGET].to_numpy(),
        }
    )

    label_counts = (
        duplicate_df
        .groupby(["group", "target"], sort=False)
        .size()
        .unstack(fill_value=0)
    )

    # Make sure both target columns exist.
    for cls in [0, 1]:
        if cls not in label_counts.columns:
            label_counts[cls] = 0

    label_counts = label_counts[[0, 1]]

    conflicting = label_counts[
        (label_counts[0] > 0) &
        (label_counts[1] > 0)
    ]

    conflicting_rows = int(conflicting.sum(axis=1).sum())

    print(f"Groups with conflicting labels: {len(conflicting)}")
    print(
        f"Rows in conflicting feature groups: "
        f"{conflicting_rows}"
    )

    if len(conflicting):
        print("\nConflicting group label counts:")
        print(conflicting.head(20).to_string())

    # ==================================================================
    # 4. Accuracy ceiling caused by contradictory feature vectors
    # ==================================================================

    all_label_counts = (
        pd.DataFrame(
            {
                "group": feature_group,
                "target": df[S.TARGET].to_numpy(),
            }
        )
        .groupby(["group", "target"], sort=False)
        .size()
        .unstack(fill_value=0)
    )

    for cls in [0, 1]:
        if cls not in all_label_counts.columns:
            all_label_counts[cls] = 0

    all_label_counts = all_label_counts[[0, 1]]

    majority_correct = int(
        all_label_counts.max(axis=1).sum()
    )

    ceiling = majority_correct / len(df)

    print("\nFeature-vector accuracy ceiling")
    print("-" * 72)
    print(
        f"Majority-within-feature-group ceiling: "
        f"{ceiling:.6f}"
    )
    print(
        f"Unavoidable disagreement rate: "
        f"{1.0 - ceiling:.6f}"
    )

    # ==================================================================
    # 5. Test ordinary StratifiedKFold for duplicate leakage
    # ==================================================================

    y = df[S.TARGET].to_numpy(dtype=int)
    indices = np.arange(len(df))

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    crossing_summary = []

    print("\nOrdinary StratifiedKFold duplicate leakage")
    print("-" * 72)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(indices, y),
        start=1,
    ):
        train_groups = set(
            feature_group.iloc[train_idx].tolist()
        )

        test_groups = set(
            feature_group.iloc[test_idx].tolist()
        )

        crossing_groups = train_groups.intersection(
            test_groups
        )

        affected_test_mask = feature_group.iloc[test_idx].isin(
            crossing_groups
        )

        affected_test_rows = int(
            affected_test_mask.sum()
        )

        crossing_summary.append(
            {
                "fold": fold,
                "crossing_groups": len(crossing_groups),
                "affected_test_rows": affected_test_rows,
            }
        )

        print(f"Fold {fold}")
        print(f"  train rows: {len(train_idx)}")
        print(f"  test rows : {len(test_idx)}")
        print(
            f"  duplicate groups crossing split: "
            f"{len(crossing_groups)}"
        )
        print(
            f"  affected test rows: "
            f"{affected_test_rows}"
        )

    total_crossing_groups = sum(
        x["crossing_groups"] for x in crossing_summary
    )

    total_affected_rows = sum(
        x["affected_test_rows"] for x in crossing_summary
    )

    # ==================================================================
    # 6. Final conclusion
    # ==================================================================

    print("\n" + "=" * 72)
    print("CONCLUSION")
    print("=" * 72)

    if total_crossing_groups > 0:
        print(
            "WARNING: feature-identical samples cross train/test "
            "boundaries under ordinary StratifiedKFold."
        )
        print(
            "The definitive benchmark should use a "
            "duplicate-group-aware split."
        )
    else:
        print(
            "No feature-identical groups crossed the tested folds."
        )
        print(
            "Ordinary StratifiedKFold does not show duplicate "
            "leakage for this dataset."
        )

    print(
        f"\nTotal fold-level crossing-group occurrences: "
        f"{total_crossing_groups}"
    )
    print(
        f"Total affected test-row occurrences: "
        f"{total_affected_rows}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()