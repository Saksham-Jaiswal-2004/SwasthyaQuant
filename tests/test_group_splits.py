import numpy as np
import pytest

from qheart.data.splits import repeated_group_stratified_folds


def test_groups_never_cross_folds():
    y = np.array(
        [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    )

    groups = np.array(
        [
            "a", "a",
            "b", "b",
            "c", "c",
            "d", "d",
            "e", "e",
            "f", "f",
        ]
    )

    folds = list(
        repeated_group_stratified_folds(
            y,
            groups,
            n_splits=3,
            n_repeats=2,
            seed=20260830,
        )
    )

    assert len(folds) == 6

    for train, test, _ in folds:
        train_groups = set(groups[train])
        test_groups = set(groups[test])

        assert train_groups.isdisjoint(test_groups)


def test_union_is_complete():
    y = np.array(
        [0, 1, 0, 1, 0, 1, 0, 1]
    )

    groups = np.array(
        ["a", "a", "b", "b", "c", "c", "d", "d"]
    )

    for train, test, _ in repeated_group_stratified_folds(
        y,
        groups,
        n_splits=2,
        n_repeats=1,
        seed=20260830,
    ):
        assert len(np.intersect1d(train, test)) == 0
        assert set(np.concatenate([train, test])) == set(range(len(y)))


def test_deterministic():
    y = np.array(
        [0, 1, 0, 1, 0, 1, 0, 1] * 3
    )

    groups = np.repeat(
        np.arange(12),
        2,
    )

    a = list(
        repeated_group_stratified_folds(
            y,
            groups,
            n_splits=3,
            n_repeats=2,
            seed=20260830,
        )
    )

    b = list(
        repeated_group_stratified_folds(
            y,
            groups,
            n_splits=3,
            n_repeats=2,
            seed=20260830,
        )
    )

    for (a_train, a_test, _), (b_train, b_test, _) in zip(a, b):
        np.testing.assert_array_equal(a_train, b_train)
        np.testing.assert_array_equal(a_test, b_test)


def test_mismatched_group_length_rejected():
    y = np.array([0, 1, 0, 1])
    groups = np.array(["a", "a", "b"])

    with pytest.raises(ValueError, match="same number"):
        list(
            repeated_group_stratified_folds(
                y,
                groups,
                n_splits=2,
                n_repeats=1,
            )
        )


def test_too_few_groups_rejected():
    y = np.array([0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b"])

    with pytest.raises(ValueError, match="at least"):
        list(
            repeated_group_stratified_folds(
                y,
                groups,
                n_splits=5,
                n_repeats=1,
            )
        )