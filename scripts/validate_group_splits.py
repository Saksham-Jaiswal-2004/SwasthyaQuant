from __future__ import annotations

import numpy as np

from qheart import schema as S
from qheart.data.loaders import load
from qheart.data.splits import repeated_group_stratified_folds


SEED = 20260830
PATH = "data/raw/cardio_train.csv"


def main() -> None:
    print("=" * 72)
    print("STAGE B.1 — REAL DATA GROUP-SPLIT VALIDATION")
    print("=" * 72)

    df = load("cardio_70000", PATH)

    y = df[S.TARGET].to_numpy(dtype=int)

    # Group on MODEL INPUT FEATURES only.
    #
    # This is deliberate: rows with identical features but different labels
    # must still remain in the same group.
    groups = np.asarray(
        __import__("pandas").util.hash_pandas_object(
            df.loc[:, S.FEATURES],
            index=False,
        )
    )

    print(f"Dataset rows: {len(df):,}")
    print(f"Unique feature groups: {len(np.unique(groups)):,}")
    print(f"Total feature groups: {len(np.unique(groups)):,}")

    folds = list(
        repeated_group_stratified_folds(
            y=y,
            groups=groups,
            n_splits=5,
            n_repeats=5,
            seed=SEED,
        )
    )

    print(f"Generated folds: {len(folds)}")

    print("\nFold statistics")
    print("-" * 72)

    all_test_counts = np.zeros(len(df), dtype=int)

    for train, test, meta in folds:
        train_groups = set(groups[train].tolist())
        test_groups = set(groups[test].tolist())

        overlap = train_groups.intersection(test_groups)

        all_test_counts[test] += 1

        print(
            f"rep={meta.rep} "
            f"fold={meta.fold} "
            f"train={meta.n_train:,} "
            f"test={meta.n_test:,} "
            f"train_prev={meta.train_prevalence:.6f} "
            f"test_prev={meta.test_prevalence:.6f} "
            f"group_overlap={len(overlap)}"
        )

        assert len(overlap) == 0

    print("\nCoverage validation")
    print("-" * 72)

    print(
        "Minimum test-fold appearances per row:",
        int(all_test_counts.min()),
    )
    print(
        "Maximum test-fold appearances per row:",
        int(all_test_counts.max()),
    )

    assert np.all(all_test_counts == 5)

    print(
        "Every row appears in exactly one test fold per repetition: PASS"
    )

    print("\n" + "=" * 72)
    print("FINAL RESULT")
    print("=" * 72)
    print("All 25 folds passed duplicate-isolation checks.")
    print("All rows have exactly five test-fold appearances.")
    print("Group-aware repeated CV is ready for benchmark integration.")
    print("=" * 72)


if __name__ == "__main__":
    main()