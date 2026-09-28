"""
STAGE B.2 — HYBRID DCQF

Paper-inspired experiment:

    classical 8 features
          +
    DCQF 24 quantum features
          |
          v
    SHAP feature ranking
          |
          v
    top-K hybrid features
          |
          v
    classical Gradient Boosting head

Everything learned by the representation, DCQF and SHAP selector is
fit inside the training fold.

This is NOT a quantum-only classifier.
DCQF is used strictly as a quantum feature extractor.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from qheart import schema as S
from qheart.data import load
from qheart.preprocess.pipeline import ClinicalRepresentation
from qheart.quantum.dcqf import DCQFExtractor


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 20260830

# Start with a pilot.
# Once this works, we will remove the cap for the final 70k benchmark.
MAX_SAMPLES = None

N_SPLITS = 5
N_REPEATS = 5

# 8 classical + 24 DCQF = 32 total.
# K=16 is fixed BEFORE looking at the test results.
TOP_K = 16

SHAP_SAMPLE = 3000

RAW_PATH = Path("data/raw/cardio_train.csv")


# ---------------------------------------------------------------------
# Group IDs
# ---------------------------------------------------------------------

def make_group_ids(df: pd.DataFrame) -> np.ndarray:
    """
    Feature-identical rows receive the same group ID.

    The target is deliberately NOT included in the group definition.
    """

    features = [
        c for c in S.FEATURES
        if c != S.TARGET
    ]

    hashed = pd.util.hash_pandas_object(
        df[features],
        index=False,
    ).to_numpy(dtype=np.uint64)

    return hashed


# ---------------------------------------------------------------------
# Group-aware fold discovery
# ---------------------------------------------------------------------

def get_group_fold_generator():
    """
    Find the group-aware repeated CV generator already implemented
    in qheart.data.splits.

    This keeps the script compatible with the current project code
    without duplicating the split algorithm.
    """

    import qheart.data.splits as splits

    candidates = [
        "repeated_stratified_group_folds",
        "repeated_group_stratified_folds",
        "group_repeated_stratified_folds",
    ]

    for name in candidates:
        fn = getattr(splits, name, None)

        if fn is None:
            continue

        sig = inspect.signature(fn)

        if "groups" in sig.parameters:
            return fn

    raise RuntimeError(
        "Could not find the group-aware repeated CV generator in "
        "qheart.data.splits. Run:\n\n"
        "    Get-Content src\\qheart\\data\\splits.py\n\n"
        "and verify that the group-aware splitter is exported."
    )


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = load(
        "cardio_70000",
        RAW_PATH,
    )

    S.validate(df)

    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        rng = np.random.default_rng(SEED)

        # Stratified pilot sample.
        pos = np.flatnonzero(df[S.TARGET].to_numpy() == 1)
        neg = np.flatnonzero(df[S.TARGET].to_numpy() == 0)

        n_pos = MAX_SAMPLES // 2
        n_neg = MAX_SAMPLES - n_pos

        selected = np.concatenate(
            [
                rng.choice(pos, size=n_pos, replace=False),
                rng.choice(neg, size=n_neg, replace=False),
            ]
        )

        selected.sort()

        df = df.iloc[selected].reset_index(drop=True)

    y = df[S.TARGET].to_numpy(dtype=int)

    groups = make_group_ids(df)

    return df, y, groups


# ---------------------------------------------------------------------
# SHAP selector
# ---------------------------------------------------------------------

def shap_rank_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rank hybrid features by mean absolute SHAP importance.

    The GradientBoosting model and SHAP explainer are fitted using
    training data only.
    """

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is required for the paper-inspired hybrid experiment.\n"
            "Install it with:\n\n"
            "    pip install shap\n"
        ) from exc

    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(SEED)

    # SHAP can become unnecessarily expensive on very large folds.
    # The SHAP ranking subset is sampled ONLY from the training fold.
    if len(X_train) > SHAP_SAMPLE:
        idx = rng.choice(
            len(X_train),
            size=SHAP_SAMPLE,
            replace=False,
        )
        X_shap = X_train[idx]
        y_shap = y_train[idx]
    else:
        X_shap = X_train
        y_shap = y_train

    selector_model = GradientBoostingClassifier(
        n_estimators=1000,
        random_state=42,
    )

    selector_model.fit(
        X_shap,
        y_shap,
    )

    explainer = shap.TreeExplainer(
        selector_model,
    )

    values = explainer.shap_values(
        X_shap,
    )

    # SHAP API differs between versions.
    if isinstance(values, list):
        values = values[-1]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, -1]

    importance = np.mean(
        np.abs(values),
        axis=0,
    )

    order = np.argsort(
        importance
    )[::-1]

    selected = order[:TOP_K]

    return selected, importance


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    p: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        roc_auc_score,
    )

    pred = (p >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn)
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp)
        else np.nan
    )

    return {
        "accuracy": accuracy_score(y_true, pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 80)
    print("STAGE B.2 — HYBRID DCQF")
    print("=" * 80)

    df, y, groups = load_dataset()

    print(f"Dataset rows: {len(df):,}")
    print(f"Positive samples: {int(y.sum()):,}")
    print(f"Negative samples: {int((1 - y).sum()):,}")
    print(f"Unique feature groups: {len(np.unique(groups)):,}")

    group_splitter = get_group_fold_generator()

    print()
    print("Configuration")
    print("-" * 80)
    print(f"CV:                 {N_SPLITS} folds × {N_REPEATS} repeats")
    print(f"Classical features: 8")
    print(f"DCQF features:      24")
    print(f"Hybrid features:    32")
    print(f"SHAP top-K:         {TOP_K}")
    print()

    # ---------------------------------------------------------------
    # Storage
    # ---------------------------------------------------------------

    results = {
        "classical": [],
        "dcqf": [],
        "hybrid_all": [],
        "hybrid_shap": [],
    }

    shap_frequency = np.zeros(
        32,
        dtype=int,
    )

    fold_number = 0

    # ---------------------------------------------------------------
    # Group-aware CV
    # ---------------------------------------------------------------

    for train_idx, test_idx, meta in group_splitter(
        y=y,
        groups=groups,
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        seed=SEED,
    ):

        fold_number += 1

        print()
        print("=" * 80)
        print(
            f"FOLD {fold_number} "
            f"(rep={meta.rep}, fold={meta.fold})"
        )
        print("=" * 80)

        X_train_raw = df.iloc[train_idx].loc[
            :,
            S.FEATURES,
        ].copy()

        X_test_raw = df.iloc[test_idx].loc[
            :,
            S.FEATURES,
        ].copy()

        y_train = y[train_idx]
        y_test = y[test_idx]

        # -----------------------------------------------------------
        # Clinical representation
        # -----------------------------------------------------------

        print("Clinical representation...")

        rep = ClinicalRepresentation(
            k=8,
            redundancy_penalty=0.5,
            output="angles",
            random_state=SEED,
        )

        X8_train = rep.fit_transform(
            X_train_raw,
            y_train,
        )

        X8_test = rep.transform(
            X_test_raw,
        )

        print(
            f"  train: {X8_train.shape}"
        )

        print(
            f"  test : {X8_test.shape}"
        )

        # -----------------------------------------------------------
        # DCQF
        # -----------------------------------------------------------

        print("DCQF...")

        dcqf = DCQFExtractor(
            orders_encoded=(2,),
            orders_read=(1, 2, 3),
            n_bins=4,
            shots=None,
            trotter_steps=1,
            dt=1.0,
            agp_scale=1.0,
            include_input=False,
            seed=SEED,
        )

        dcqf.fit(
            X8_train,
            y_train,
        )

        Q_train = dcqf.transform(
            X8_train,
        )

        Q_test = dcqf.transform(
            X8_test,
        )

        print(
            f"  train quantum: {Q_train.shape}"
        )

        print(
            f"  test quantum : {Q_test.shape}"
        )

        # -----------------------------------------------------------
        # Hybrid representation
        # -----------------------------------------------------------

        H_train = np.concatenate(
            [
                X8_train,
                Q_train,
            ],
            axis=1,
        )

        H_test = np.concatenate(
            [
                X8_test,
                Q_test,
            ],
            axis=1,
        )

        feature_names = (
            [f"classical_{i}" for i in range(8)]
            +
            [
                f"quantum_{name}"
                for name in dcqf.feature_names()
            ]
        )

        assert H_train.shape[1] == 32
        assert H_test.shape[1] == 32

        # -----------------------------------------------------------
        # Standardisation
        # -----------------------------------------------------------

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()

        H_train_scaled = scaler.fit_transform(
            H_train,
        )

        H_test_scaled = scaler.transform(
            H_test,
        )

        # -----------------------------------------------------------
        # Classical baseline
        # -----------------------------------------------------------

        from sklearn.ensemble import GradientBoostingClassifier

        classical_model = GradientBoostingClassifier(
            n_estimators=1000,
            random_state=42,
        )

        classical_model.fit(
            H_train_scaled[:, :8],
            y_train,
        )

        p_classical = classical_model.predict_proba(
            H_test_scaled[:, :8]
        )[:, 1]

        results["classical"].append(
            evaluate_predictions(
                y_test,
                p_classical,
            )
        )

        # -----------------------------------------------------------
        # DCQF-only
        # -----------------------------------------------------------

        dcqf_model = GradientBoostingClassifier(
            n_estimators=1000,
            random_state=42,
        )

        dcqf_model.fit(
            H_train_scaled[:, 8:],
            y_train,
        )

        p_dcqf = dcqf_model.predict_proba(
            H_test_scaled[:, 8:]
        )[:, 1]

        results["dcqf"].append(
            evaluate_predictions(
                y_test,
                p_dcqf,
            )
        )

        # -----------------------------------------------------------
        # Hybrid — all 32 features
        # -----------------------------------------------------------

        hybrid_all = GradientBoostingClassifier(
            n_estimators=1000,
            random_state=42,
        )

        hybrid_all.fit(
            H_train_scaled,
            y_train,
        )

        p_hybrid_all = hybrid_all.predict_proba(
            H_test_scaled
        )[:, 1]

        results["hybrid_all"].append(
            evaluate_predictions(
                y_test,
                p_hybrid_all,
            )
        )

        # -----------------------------------------------------------
        # SHAP selection
        # -----------------------------------------------------------

        print("SHAP feature selection...")

        selected, importance = shap_rank_features(
            H_train_scaled,
            y_train,
            feature_names,
        )

        shap_frequency[selected] += 1

        print(
            "  Selected features:"
        )

        for rank, idx in enumerate(
            selected,
            start=1,
        ):
            print(
                f"    {rank:2d}. "
                f"{feature_names[idx]:25s} "
                f"importance={importance[idx]:.6f}"
            )

        H_train_selected = H_train_scaled[
            :,
            selected,
        ]

        H_test_selected = H_test_scaled[
            :,
            selected,
        ]

        hybrid_shap = GradientBoostingClassifier(
            n_estimators=1000,
            random_state=42,
        )

        hybrid_shap.fit(
            H_train_selected,
            y_train,
        )

        p_hybrid_shap = hybrid_shap.predict_proba(
            H_test_selected
        )[:, 1]

        results["hybrid_shap"].append(
            evaluate_predictions(
                y_test,
                p_hybrid_shap,
            )
        )

        # -----------------------------------------------------------
        # Fold summary
        # -----------------------------------------------------------

        print()
        print("Fold results")
        print("-" * 80)

        for name in results:
            r = results[name][-1]

            print(
                f"{name:15s} "
                f"Acc={r['accuracy']:.4f} "
                f"ROC-AUC={r['roc_auc']:.4f} "
                f"PR-AUC={r['pr_auc']:.4f}"
            )

    # -----------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(
        f"{'Model':18s}"
        f"{'Acc':>10s}"
        f"{'Sens':>10s}"
        f"{'Spec':>10s}"
        f"{'ROC-AUC':>10s}"
        f"{'PR-AUC':>10s}"
    )

    print("-" * 80)

    for name, rows in results.items():

        print(
            f"{name:18s}"
            f"{np.mean([r['accuracy'] for r in rows]):10.4f}"
            f"{np.mean([r['sensitivity'] for r in rows]):10.4f}"
            f"{np.mean([r['specificity'] for r in rows]):10.4f}"
            f"{np.mean([r['roc_auc'] for r in rows]):10.4f}"
            f"{np.mean([r['pr_auc'] for r in rows]):10.4f}"
        )

    # -----------------------------------------------------------------
    # SHAP stability
    # -----------------------------------------------------------------

    print()
    print("=" * 80)
    print("SHAP FEATURE SELECTION STABILITY")
    print("=" * 80)

    names = (
        [f"classical_{i}" for i in range(8)]
        +
        [
            f"quantum_{name}"
            for name in dcqf.feature_names()
        ]
    )

    for idx in np.argsort(
        shap_frequency
    )[::-1]:

        if shap_frequency[idx] == 0:
            continue

        print(
            f"{names[idx]:25s} "
            f"selected {shap_frequency[idx]}/{fold_number} folds"
        )

    # -----------------------------------------------------------------
    # Key comparison
    # -----------------------------------------------------------------

    def mean_metric(model, metric):
        return float(
            np.mean(
                [
                    r[metric]
                    for r in results[model]
                ]
            )
        )

    classical_pr = mean_metric(
        "classical",
        "pr_auc",
    )

    dcqf_pr = mean_metric(
        "dcqf",
        "pr_auc",
    )

    hybrid_all_pr = mean_metric(
        "hybrid_all",
        "pr_auc",
    )

    hybrid_shap_pr = mean_metric(
        "hybrid_shap",
        "pr_auc",
    )

    print()
    print("=" * 80)
    print("HYBRID DELTAS")
    print("=" * 80)

    print(
        f"DCQF - classical PR-AUC: "
        f"{dcqf_pr - classical_pr:+.4f}"
    )

    print(
        f"Hybrid-all - classical PR-AUC: "
        f"{hybrid_all_pr - classical_pr:+.4f}"
    )

    print(
        f"Hybrid-SHAP - classical PR-AUC: "
        f"{hybrid_shap_pr - classical_pr:+.4f}"
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  Hybrid-SHAP > classical:"
        " quantum features provide complementary predictive information."
    )

    print(
        "  Hybrid-SHAP > Hybrid-all:"
        " feature selection removes redundant/noisy dimensions."
    )

    print(
        "  DCQF < classical but Hybrid-SHAP > classical:"
        " quantum information is useful primarily in combination with"
        " the classical clinical representation."
    )

    print()
    print(
        "DCQF provenance:"
    )
    print(
        dcqf.provenance()
    )


if __name__ == "__main__":
    main()