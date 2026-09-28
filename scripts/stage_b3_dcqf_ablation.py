"""
STAGE B.3 — DCQF FEATURE ABLATION

Goal
----
Determine which parts of the 24-dimensional DCQF representation
actually contribute useful predictive information.

Representations compared:

    classical
        8 clinical features

    quantum_singles
        Z0 ... Z7

    quantum_pairs
        Z0Z1 ... Z7Z0

    quantum_triples
        Z0Z1Z2 ... Z7Z0Z1

    hybrid_singles
        classical + singles

    hybrid_pairs
        classical + singles + pairs

    hybrid_all
        classical + all 24 DCQF features

Everything learned is fitted inside each group-aware CV training fold.

This experiment does NOT claim quantum advantage.
It is a representation ablation.
"""

from __future__ import annotations

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

MAX_SAMPLES = None

N_SPLITS = 5
N_REPEATS = 5

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
# Group-aware CV
# ---------------------------------------------------------------------

def get_group_fold_generator():
    """
    Find the group-aware repeated CV generator already implemented
    in qheart.data.splits.
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
        "qheart.data.splits."
    )


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:

    df = load(
        "cardio_70000",
        RAW_PATH,
    )

    S.validate(df)

    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:

        rng = np.random.default_rng(SEED)

        pos = np.flatnonzero(
            df[S.TARGET].to_numpy() == 1
        )

        neg = np.flatnonzero(
            df[S.TARGET].to_numpy() == 0
        )

        n_pos = MAX_SAMPLES // 2
        n_neg = MAX_SAMPLES - n_pos

        selected = np.concatenate(
            [
                rng.choice(
                    pos,
                    size=n_pos,
                    replace=False,
                ),
                rng.choice(
                    neg,
                    size=n_neg,
                    replace=False,
                ),
            ]
        )

        selected.sort()

        df = df.iloc[selected].reset_index(drop=True)

    y = df[S.TARGET].to_numpy(dtype=int)

    groups = make_group_ids(df)

    return df, y, groups


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

    pred = (
        p >= threshold
    ).astype(int)

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
        "accuracy": accuracy_score(
            y_true,
            pred,
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc_score(
            y_true,
            p,
        ),
        "pr_auc": average_precision_score(
            y_true,
            p,
        ),
    }


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

def fit_predict(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    model = GradientBoostingClassifier(
        n_estimators=1000,
        random_state=42,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )

    return model.predict_proba(
        X_test_scaled
    )[:, 1]


# ---------------------------------------------------------------------
# Clinical feature mapping
# ---------------------------------------------------------------------

def describe_representation(rep) -> None:
    """
    Try to recover the names/order of the 8 clinical representation
    features without assuming a particular internal attribute name.
    """

    print()
    print("=" * 80)
    print("CLINICAL REPRESENTATION MAPPING")
    print("=" * 80)

    candidates = [
        "feature_names_",
        "selected_features_",
        "selected_columns_",
        "columns_",
        "features_",
        "selected_",
    ]

    found = False

    for attr in candidates:

        value = getattr(rep, attr, None)

        if value is None:
            continue

        if callable(value):
            try:
                value = value()
            except Exception:
                continue

        try:
            values = list(value)
        except Exception:
            continue

        if len(values) != 8:
            continue

        print(f"Representation attribute: {attr}")

        for i, name in enumerate(values):
            print(
                f"  classical_{i} -> {name}"
            )

        print()
        print("Corresponding DCQF qubit axes:")

        for i, name in enumerate(values):
            print(
                f"  Z{i} -> classical_{i} -> {name}"
            )

        found = True
        break

    if not found:

        print(
            "Could not automatically identify the representation "
            "feature-name attribute."
        )

        print()
        print(
            "Available representation attributes:"
        )

        for key in sorted(
            rep.__dict__.keys()
        ):
            if (
                "feature" in key.lower()
                or "select" in key.lower()
                or "column" in key.lower()
                or "name" in key.lower()
            ):
                print(
                    f"  {key}: "
                    f"{type(getattr(rep, key)).__name__}"
                )

        print()
        print(
            "The ablation itself is unaffected."
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 80)
    print("STAGE B.3 — DCQF FEATURE ABLATION")
    print("=" * 80)

    df, y, groups = load_dataset()

    print(
        f"Dataset rows: {len(df):,}"
    )

    print(
        f"Positive samples: {int(y.sum()):,}"
    )

    print(
        f"Negative samples: {int((1 - y).sum()):,}"
    )

    print(
        f"Unique feature groups: "
        f"{len(np.unique(groups)):,}"
    )

    print()

    print("Configuration")
    print("-" * 80)

    print(
        f"CV:                 "
        f"{N_SPLITS} folds × {N_REPEATS} repeats"
    )

    print(
        "Clinical features:  8"
    )

    print(
        "DCQF features:      24"
    )

    print(
        "Quantum singles:    8"
    )

    print(
        "Quantum pairs:      8"
    )

    print(
        "Quantum triples:    8"
    )

    print()

    # ---------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------

    model_names = [
        "classical",
        "quantum_singles",
        "quantum_pairs",
        "quantum_triples",
        "hybrid_singles",
        "hybrid_pairs",
        "hybrid_all",
    ]

    results = {
        name: []
        for name in model_names
    }

    # ---------------------------------------------------------------
    # Feature stability / frequency
    # ---------------------------------------------------------------

    dcqf_names = None

    # ---------------------------------------------------------------
    # Group-aware CV
    # ---------------------------------------------------------------

    group_splitter = get_group_fold_generator()

    fold_number = 0

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

        X_train_raw = df.iloc[
            train_idx
        ].loc[
            :,
            S.FEATURES,
        ].copy()

        X_test_raw = df.iloc[
            test_idx
        ].loc[
            :,
            S.FEATURES,
        ].copy()

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        # -----------------------------------------------------------
        # Clinical representation
        # -----------------------------------------------------------

        print(
            "Clinical representation..."
        )

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

        # Print mapping once.
        if fold_number == 1:
            describe_representation(
                rep
            )

        # -----------------------------------------------------------
        # DCQF
        # -----------------------------------------------------------

        print(
            "DCQF..."
        )

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
            X8_train
        )

        Q_test = dcqf.transform(
            X8_test
        )

        if dcqf_names is None:
            dcqf_names = dcqf.feature_names()

            print()
            print(
                "DCQF feature layout:"
            )

            for i, name in enumerate(
                dcqf_names
            ):
                print(
                    f"  {i:2d}: {name}"
                )

        print(
            f"  train quantum: "
            f"{Q_train.shape}"
        )

        print(
            f"  test quantum : "
            f"{Q_test.shape}"
        )

        # -----------------------------------------------------------
        # Feature blocks
        # -----------------------------------------------------------

        classical_train = X8_train
        classical_test = X8_test

        quantum_singles_train = Q_train[:, 0:8]
        quantum_singles_test = Q_test[:, 0:8]

        quantum_pairs_train = Q_train[:, 8:16]
        quantum_pairs_test = Q_test[:, 8:16]

        quantum_triples_train = Q_train[:, 16:24]
        quantum_triples_test = Q_test[:, 16:24]

        hybrid_singles_train = np.concatenate(
            [
                classical_train,
                quantum_singles_train,
            ],
            axis=1,
        )

        hybrid_singles_test = np.concatenate(
            [
                classical_test,
                quantum_singles_test,
            ],
            axis=1,
        )

        hybrid_pairs_train = np.concatenate(
            [
                classical_train,
                quantum_singles_train,
                quantum_pairs_train,
            ],
            axis=1,
        )

        hybrid_pairs_test = np.concatenate(
            [
                classical_test,
                quantum_singles_test,
                quantum_pairs_test,
            ],
            axis=1,
        )

        hybrid_all_train = np.concatenate(
            [
                classical_train,
                Q_train,
            ],
            axis=1,
        )

        hybrid_all_test = np.concatenate(
            [
                classical_test,
                Q_test,
            ],
            axis=1,
        )

        # -----------------------------------------------------------
        # Run ablations
        # -----------------------------------------------------------

        feature_sets = {

            "classical": (
                classical_train,
                classical_test,
            ),

            "quantum_singles": (
                quantum_singles_train,
                quantum_singles_test,
            ),

            "quantum_pairs": (
                quantum_pairs_train,
                quantum_pairs_test,
            ),

            "quantum_triples": (
                quantum_triples_train,
                quantum_triples_test,
            ),

            "hybrid_singles": (
                hybrid_singles_train,
                hybrid_singles_test,
            ),

            "hybrid_pairs": (
                hybrid_pairs_train,
                hybrid_pairs_test,
            ),

            "hybrid_all": (
                hybrid_all_train,
                hybrid_all_test,
            ),
        }

        for name in model_names:

            Xtr, Xte = feature_sets[
                name
            ]

            print(
                f"  fitting {name:20s} "
                f"features={Xtr.shape[1]}"
            )

            p = fit_predict(
                Xtr,
                Xte,
                y_train,
            )

            results[name].append(
                evaluate_predictions(
                    y_test,
                    p,
                )
            )

        # -----------------------------------------------------------
        # Fold summary
        # -----------------------------------------------------------

        print()

        print(
            "Fold results"
        )

        print(
            "-" * 80
        )

        for name in model_names:

            r = results[name][-1]

            print(
                f"{name:20s} "
                f"Acc={r['accuracy']:.4f} "
                f"Sens={r['sensitivity']:.4f} "
                f"Spec={r['specificity']:.4f} "
                f"ROC-AUC={r['roc_auc']:.4f} "
                f"PR-AUC={r['pr_auc']:.4f}"
            )

    # ---------------------------------------------------------------
    # Final results
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL DCQF ABLATION RESULTS")
    print("=" * 80)

    print(
        f"{'Model':22s}"
        f"{'Feat':>6s}"
        f"{'Acc':>9s}"
        f"{'Sens':>9s}"
        f"{'Spec':>9s}"
        f"{'ROC-AUC':>10s}"
        f"{'PR-AUC':>9s}"
    )

    print("-" * 80)

    feature_counts = {
        "classical": 8,
        "quantum_singles": 8,
        "quantum_pairs": 8,
        "quantum_triples": 8,
        "hybrid_singles": 16,
        "hybrid_pairs": 24,
        "hybrid_all": 32,
    }

    for name in model_names:

        rows = results[name]

        print(
            f"{name:22s}"
            f"{feature_counts[name]:6d}"
            f"{np.mean([r['accuracy'] for r in rows]):9.4f}"
            f"{np.mean([r['sensitivity'] for r in rows]):9.4f}"
            f"{np.mean([r['specificity'] for r in rows]):9.4f}"
            f"{np.mean([r['roc_auc'] for r in rows]):10.4f}"
            f"{np.mean([r['pr_auc'] for r in rows]):9.4f}"
        )

    # ---------------------------------------------------------------
    # Delta analysis
    # ---------------------------------------------------------------

    def mean_metric(
        model: str,
        metric: str,
    ) -> float:

        return float(
            np.mean(
                [
                    r[metric]
                    for r in results[model]
                ]
            )
        )

    classical_roc = mean_metric(
        "classical",
        "roc_auc",
    )

    classical_pr = mean_metric(
        "classical",
        "pr_auc",
    )

    print()
    print("=" * 80)
    print("DELTA ANALYSIS")
    print("=" * 80)

    for name in model_names:

        if name == "classical":
            continue

        roc = mean_metric(
            name,
            "roc_auc",
        )

        pr = mean_metric(
            name,
            "pr_auc",
        )

        print(
            f"{name:22s} "
            f"ROC-AUC Δ={roc - classical_roc:+.4f} "
            f"PR-AUC Δ={pr - classical_pr:+.4f}"
        )

    # ---------------------------------------------------------------
    # Hierarchical quantum contribution
    # ---------------------------------------------------------------

    print()
    print("=" * 80)
    print("HIERARCHICAL QUANTUM CONTRIBUTION")
    print("=" * 80)

    single_pr = mean_metric(
        "hybrid_singles",
        "pr_auc",
    )

    pair_pr = mean_metric(
        "hybrid_pairs",
        "pr_auc",
    )

    all_pr = mean_metric(
        "hybrid_all",
        "pr_auc",
    )

    print(
        f"Classical PR-AUC:          "
        f"{classical_pr:.4f}"
    )

    print(
        f"+ quantum singles:         "
        f"{single_pr:.4f} "
        f"(Δ={single_pr - classical_pr:+.4f})"
    )

    print(
        f"+ quantum pairs:           "
        f"{pair_pr:.4f} "
        f"(Δ={pair_pr - single_pr:+.4f})"
    )

    print(
        f"+ quantum triples:        "
        f"{all_pr:.4f} "
        f"(Δ={all_pr - pair_pr:+.4f})"
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "  This experiment determines whether the "
        "useful DCQF contribution is concentrated "
        "in single-, pair-, or three-body observables."
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