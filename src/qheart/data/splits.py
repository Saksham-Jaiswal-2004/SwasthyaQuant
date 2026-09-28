"""Splitting. Pure numpy, deterministic from one base seed, and self-checking.

Implemented here rather than imported from sklearn for three reasons: the
assertions below (which caught real bugs), the ``meta`` record that makes every
fold traceable in the results ledger, and the fact that the split logic is the
one place a silent bug is both easy to introduce and impossible to notice from
the metrics -- a leaky split makes numbers better, not worse.

Rules this module enforces, on every call:
  * train and test never intersect;
  * their union is the whole dataset (nothing silently dropped);
  * every fold contains both classes, or you get an exception rather than a
    degenerate ROC-AUC of nan that quietly propagates into a mean;
  * group-aware folds never split identical groups between train and test.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator

import numpy as np

from qheart.seeds import spawn

__all__ = [
    "FoldMeta",
    "repeated_stratified_folds",
    "repeated_group_stratified_folds",
    "holdout_split",
    "cohort_holdout",
]


@dataclass(frozen=True)
class FoldMeta:
    rep: int
    fold: int
    n_train: int
    n_test: int
    train_prevalence: float
    test_prevalence: float

    def as_dict(self) -> dict:
        return asdict(self)


def _check(
    train: np.ndarray,
    test: np.ndarray,
    y: np.ndarray,
    where: str,
) -> None:
    """Validate basic train/test invariants."""
    if np.intersect1d(train, test).size:
        raise AssertionError(
            f"{where}: train/test overlap -- leakage"
        )

    if train.size + test.size != y.size:
        raise AssertionError(
            f"{where}: {train.size}+{test.size} != {y.size} rows"
        )

    for name, idx in (("train", train), ("test", test)):
        classes = np.unique(y[idx])

        if classes.size < 2:
            raise AssertionError(
                f"{where}: {name} fold has only class "
                f"{classes.tolist()}. "
                f"Reduce n_splits or check the class balance -- "
                f"do not paper over this, it makes AUC undefined."
            )


def _stratified_chunks(
    y: np.ndarray,
    n_splits: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Partition indices into n_splits parts, preserving class ratios."""
    parts: list[list[np.ndarray]] = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)

        # array_split spreads the remainder over the first chunks, so sizes
        # differ by at most one -- the same guarantee sklearn gives.
        for j, chunk in enumerate(np.array_split(idx, n_splits)):
            parts[j].append(chunk)

    return [
        np.sort(np.concatenate(p))
        for p in parts
    ]


def _group_stratified_chunks(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Partition whole groups into approximately stratified folds.

    Groups are never split across folds.

    Assignment is greedy and deterministic. Larger groups are assigned first,
    while the objective simultaneously considers:
      * total fold size;
      * positive-class count;
      * negative-class count.

    This is appropriate for duplicate-aware evaluation, where identical
    feature vectors must never occur in both train and test.
    """
    y = np.asarray(y).ravel()
    groups = np.asarray(groups)

    if y.size == 0:
        raise ValueError("empty y")

    if groups.size != y.size:
        raise ValueError(
            "y and groups must have the same length"
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be >= 2"
        )

    unique_groups, inverse = np.unique(
        groups,
        return_inverse=True,
    )

    n_groups = unique_groups.size

    if n_groups < n_splits:
        raise ValueError(
            f"need at least {n_splits} groups, got {n_groups}"
        )

    classes = np.unique(y)

    if classes.size != 2:
        raise ValueError(
            "group stratification currently requires exactly two classes"
        )

    # --------------------------------------------------------------
    # Per-group statistics
    # --------------------------------------------------------------

    group_size = np.bincount(
        inverse,
        minlength=n_groups,
    ).astype(float)

    group_pos = np.bincount(
        inverse,
        weights=(y == classes[1]).astype(float),
        minlength=n_groups,
    )

    group_neg = group_size - group_pos

    # --------------------------------------------------------------
    # Assignment order
    #
    # Larger / unusual groups constrain the solution more strongly,
    # so place them first.
    # --------------------------------------------------------------

    prevalence = group_pos / np.maximum(
        group_size,
        1.0,
    )

    order = np.arange(n_groups)

    # Randomization is deterministic because rng comes from spawn().
    rng.shuffle(order)

    # Stable deterministic ordering:
    #   1. larger groups first
    #   2. higher prevalence first
    order = order[
        np.lexsort(
            (
                -prevalence[order],
                -group_size[order],
            )
        )
    ]

    # --------------------------------------------------------------
    # Target fold statistics
    # --------------------------------------------------------------

    total_n = float(y.size)
    total_pos = float(np.sum(y == classes[1]))
    total_neg = float(np.sum(y == classes[0]))

    target_n = total_n / n_splits
    target_pos = total_pos / n_splits
    target_neg = total_neg / n_splits

    fold_n = np.zeros(
        n_splits,
        dtype=float,
    )

    fold_pos = np.zeros(
        n_splits,
        dtype=float,
    )

    fold_neg = np.zeros(
        n_splits,
        dtype=float,
    )

    assignments = np.full(
        n_groups,
        -1,
        dtype=int,
    )

    # --------------------------------------------------------------
    # Objective
    # --------------------------------------------------------------

    def objective(
        n: np.ndarray,
        pos: np.ndarray,
        neg: np.ndarray,
    ) -> float:
        """Normalized squared imbalance across folds."""
        size_error = np.sum(
            ((n - target_n) / max(target_n, 1.0)) ** 2
        )

        pos_error = np.sum(
            ((pos - target_pos) / max(target_pos, 1.0)) ** 2
        )

        neg_error = np.sum(
            ((neg - target_neg) / max(target_neg, 1.0)) ** 2
        )

        return float(
            size_error +
            pos_error +
            neg_error
        )

    # --------------------------------------------------------------
    # Greedy whole-group assignment
    # --------------------------------------------------------------

    for group_idx in order:
        best_fold = None
        best_score = None

        # Deterministic randomized tie-breaking.
        candidate_folds = np.arange(n_splits)
        rng.shuffle(candidate_folds)

        for fold in candidate_folds:
            trial_n = fold_n.copy()
            trial_pos = fold_pos.copy()
            trial_neg = fold_neg.copy()

            trial_n[fold] += group_size[group_idx]
            trial_pos[fold] += group_pos[group_idx]
            trial_neg[fold] += group_neg[group_idx]

            score = objective(
                trial_n,
                trial_pos,
                trial_neg,
            )

            if best_score is None or score < best_score:
                best_score = score
                best_fold = int(fold)

        assert best_fold is not None

        assignments[group_idx] = best_fold

        fold_n[best_fold] += group_size[group_idx]
        fold_pos[best_fold] += group_pos[group_idx]
        fold_neg[best_fold] += group_neg[group_idx]

    # --------------------------------------------------------------
    # Convert group assignments back to row indices
    # --------------------------------------------------------------

    chunks = [
        np.sort(
            np.flatnonzero(
                assignments[inverse] == fold
            )
        )
        for fold in range(n_splits)
    ]

    return chunks


def repeated_stratified_folds(
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 20260830,
) -> Iterator[
    tuple[np.ndarray, np.ndarray, FoldMeta]
]:
    """Yield train/test/meta for repeated ordinary stratified folds.

    Defaults give 25 fits per model, which is the point: with a few hundred
    test rows per fold, a single split cannot distinguish a real 1-point gain
    from noise. Report mean and standard deviation across all 25.
    """
    y = np.asarray(y).ravel()

    if y.size == 0:
        raise ValueError("empty y")

    if n_splits < 2:
        raise ValueError(
            "n_splits must be >= 2"
        )

    all_idx = np.arange(y.size)

    for rep in range(n_repeats):
        rng = spawn(
            seed,
            "split",
            n_splits,
            rep,
        )

        chunks = _stratified_chunks(
            y,
            n_splits,
            rng,
        )

        for fold, test in enumerate(chunks):
            train = np.setdiff1d(
                all_idx,
                test,
                assume_unique=False,
            )

            _check(
                train,
                test,
                y,
                f"rep{rep}/fold{fold}",
            )

            yield train, test, FoldMeta(
                rep=rep,
                fold=fold,
                n_train=int(train.size),
                n_test=int(test.size),
                train_prevalence=float(
                    y[train].mean()
                ),
                test_prevalence=float(
                    y[test].mean()
                ),
            )


def repeated_group_stratified_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 20260830,
) -> Iterator[
    tuple[np.ndarray, np.ndarray, FoldMeta]
]:
    """Yield repeated stratified folds without splitting groups.

    ``groups[i]`` identifies samples that must remain in the same fold.

    Stratification is approximate because entire groups are assigned as
    indivisible units.

    This splitter is intended for duplicate-aware evaluation where identical
    feature vectors must never occur in both train and test.
    """
    y = np.asarray(y).ravel()
    groups = np.asarray(groups)

    if y.size == 0:
        raise ValueError("empty y")

    if groups.size != y.size:
        raise ValueError(
            "groups must have the same number of rows as y"
        )

    if n_splits < 2:
        raise ValueError(
            "n_splits must be >= 2"
        )

    all_idx = np.arange(y.size)

    for rep in range(n_repeats):
        rng = spawn(
            seed,
            "group_split",
            n_splits,
            rep,
        )

        chunks = _group_stratified_chunks(
            y=y,
            groups=groups,
            n_splits=n_splits,
            rng=rng,
        )

        for fold, test in enumerate(chunks):
            train = np.setdiff1d(
                all_idx,
                test,
                assume_unique=False,
            )

            _check(
                train,
                test,
                y,
                f"group_rep{rep}/fold{fold}",
            )

            # Explicit group-isolation check.
            train_groups = np.unique(
                groups[train]
            )

            test_groups = np.unique(
                groups[test]
            )

            if np.intersect1d(
                train_groups,
                test_groups,
            ).size:
                raise AssertionError(
                    f"group_rep{rep}/fold{fold}: "
                    "group appears in both train and test"
                )

            yield train, test, FoldMeta(
                rep=rep,
                fold=fold,
                n_train=int(train.size),
                n_test=int(test.size),
                train_prevalence=float(
                    y[train].mean()
                ),
                test_prevalence=float(
                    y[test].mean()
                ),
            )


def holdout_split(
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 20260830,
) -> tuple[np.ndarray, np.ndarray]:
    """A single stratified holdout, sealed until the very end.

    Use for exactly one thing: the final locked evaluation in Week 4. All model
    selection and tuning happens inside ``repeated_stratified_folds`` on the
    training portion. Touch this split more than once and it stops being a test
    set.
    """
    y = np.asarray(y).ravel()

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            "test_size must be in (0, 1)"
        )

    rng = spawn(
        seed,
        "holdout",
        test_size,
    )

    test_parts = []

    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)

        k = int(
            round(test_size * idx.size)
        )

        k = min(
            max(k, 1),
            idx.size - 1,
        )

        test_parts.append(
            idx[:k]
        )

    test = np.sort(
        np.concatenate(test_parts)
    )

    train = np.setdiff1d(
        np.arange(y.size),
        test,
    )

    _check(
        train,
        test,
        y,
        "holdout",
    )

    return train, test


def cohort_holdout(
    cohort: np.ndarray,
    y: np.ndarray,
    test_cohorts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Train on some source hospitals, test on others -- true external validation.

    Only usable with the ``heart_uci_cohorts`` loader. The Kaggle 918 merge
    dropped the cohort column, so calling this on that source raises. This is
    the strongest generalization evidence available for this problem:
    performance across sites typically drops a lot, and reporting that
    honestly is far more persuasive than a high pooled number.
    """
    cohort = np.asarray(
        cohort,
        dtype=object,
    )

    if all(
        c is None or c != c
        for c in cohort
    ):
        raise ValueError(
            "cohort is unavailable for this source "
            "(the Kaggle merge dropped it). "
            "Use loader 'heart_uci_cohorts' for external validation, "
            "or report subgroup metrics instead and label them as such."
        )

    test = np.sort(
        np.flatnonzero(
            np.isin(
                cohort,
                test_cohorts,
            )
        )
    )

    train = np.setdiff1d(
        np.arange(cohort.size),
        test,
    )

    if test.size == 0:
        raise ValueError(
            f"no rows matched test_cohorts={test_cohorts}"
        )

    _check(
        train,
        test,
        np.asarray(y).ravel(),
        f"cohort_holdout{test_cohorts}",
    )

    return train, test