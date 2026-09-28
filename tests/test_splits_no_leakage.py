"""The split tests. A leaky split inflates every downstream number silently,
so these assertions matter more than any model test in the suite.
"""

import numpy as np
import pytest

from qheart.data.splits import cohort_holdout, holdout_split, repeated_stratified_folds


@pytest.fixture
def y():
    return np.repeat([1, 0], [508, 410])          # roughly the real prevalence


def test_folds_partition_exactly_and_never_overlap(y):
    seen_test = []
    n = 0
    for tr, te, meta in repeated_stratified_folds(y, n_splits=5, n_repeats=2, seed=1):
        assert np.intersect1d(tr, te).size == 0
        assert tr.size + te.size == y.size
        seen_test.append(te)
        n += 1
        assert meta.n_train == tr.size and meta.n_test == te.size
    assert n == 10
    # Within one repeat the five test folds must tile the dataset exactly once.
    first_repeat = np.sort(np.concatenate(seen_test[:5]))
    assert np.array_equal(first_repeat, np.arange(y.size))


def test_stratification_holds_in_every_fold(y):
    overall = y.mean()
    for tr, te, meta in repeated_stratified_folds(y, n_splits=5, n_repeats=1, seed=2):
        assert meta.test_prevalence == pytest.approx(overall, abs=0.02)
        assert meta.train_prevalence == pytest.approx(overall, abs=0.02)


def test_same_seed_reproduces_and_different_seed_does_not(y):
    a = [te.tolist() for _, te, _ in repeated_stratified_folds(y, seed=42, n_repeats=1)]
    b = [te.tolist() for _, te, _ in repeated_stratified_folds(y, seed=42, n_repeats=1)]
    c = [te.tolist() for _, te, _ in repeated_stratified_folds(y, seed=43, n_repeats=1)]
    assert a == b, "a config must reproduce its folds exactly"
    assert a != c, "different seeds must actually give different folds"


def test_repeats_are_not_identical(y):
    folds = list(repeated_stratified_folds(y, n_splits=5, n_repeats=2, seed=3))
    assert folds[0][1].tolist() != folds[5][1].tolist(), \
        "repeat 2 reshuffles, otherwise 25 fits give 5 folds' worth of information"


def test_degenerate_fold_raises_rather_than_returning_nan():
    y = np.array([1] + [0] * 30)          # one positive cannot be split 5 ways
    with pytest.raises(AssertionError, match="only class"):
        list(repeated_stratified_folds(y, n_splits=5, n_repeats=1, seed=4))


def test_holdout_is_stratified_and_disjoint(y):
    tr, te = holdout_split(y, test_size=0.2, seed=5)
    assert np.intersect1d(tr, te).size == 0
    assert tr.size + te.size == y.size
    assert te.size == pytest.approx(0.2 * y.size, abs=2)
    assert y[te].mean() == pytest.approx(y.mean(), abs=0.02)


def test_cohort_holdout_rejects_the_kaggle_source(y):
    with pytest.raises(ValueError, match="cohort is unavailable"):
        cohort_holdout(np.full(y.size, np.nan, dtype=object), y, ["hungarian"])


def test_cohort_holdout_splits_by_site():
    cohort = np.array(["cleveland"] * 60 + ["hungarian"] * 40, dtype=object)
    y = np.r_[np.repeat([1, 0], [30, 30]), np.repeat([1, 0], [20, 20])]
    tr, te = cohort_holdout(cohort, y, ["hungarian"])
    assert set(cohort[te]) == {"hungarian"} and set(cohort[tr]) == {"cleveland"}
