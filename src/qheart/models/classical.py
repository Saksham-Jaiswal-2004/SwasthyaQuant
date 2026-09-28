"""Classical baselines.

Five strong classical reference models. Every model is a complete
scikit-learn Pipeline, with the fold-safe clinical representation inside
the pipeline.

The representation performs:

    raw cardiovascular data
        -> deterministic clinical feature engineering
        -> MI/mRMR feature selection
        -> training-fold imputation
        -> training-fold standardisation
        -> classifier

Because preprocessing is structurally part of the Pipeline, fitting a
model inside cross-validation cannot accidentally fit the learned
preprocessing steps on the test fold.

The RBF baseline uses a Nyström approximation rather than
SVC(probability=True). This keeps the nonlinear classical reference
computationally practical for the 70,000-row cardiovascular dataset
while preserving the RBF-kernel comparison.
"""

from __future__ import annotations

__all__ = [
    "logreg",
    "svm_rbf",
    "random_forest",
    "xgboost",
    "mlp",
    "CLASSICAL",
]

SEED = 20260830

N_FEATURES = 8
REDUNDANCY_PENALTY = 0.5


def _pipe(
    model,
    *,
    k: int = N_FEATURES,
    redundancy_penalty: float = REDUNDANCY_PENALTY,
):
    """Build a fold-safe clinical -> classifier Pipeline."""

    from sklearn.pipeline import Pipeline

    from qheart.preprocess.pipeline import ClinicalRepresentation

    representation = ClinicalRepresentation(
        k=k,
        redundancy_penalty=redundancy_penalty,
        output="standard",
        random_state=SEED,
    )

    return Pipeline(
        [
            (
                "prep",
                representation,
            ),
            (
                "model",
                model,
            ),
        ]
    )


def logreg(
    C: float = 1.0,
    **kw,
):
    """Regularised logistic-regression reference."""

    from sklearn.linear_model import LogisticRegression

    return _pipe(
        LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
            **kw,
        )
    )


def svm_rbf(
    C: float = 1.0,
    gamma=None,
    n_components: int = 512,
    **kw,
):
    """Scalable nonlinear RBF reference.

    Uses a Nyström approximation of the RBF kernel followed by
    logistic regression. The entire transformation remains inside
    the fold-local model pipeline.

    ``gamma=None`` lets Nystroem use its default RBF bandwidth.
    ``n_components`` controls the size of the kernel approximation.
    """

    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    rbf_features = Nystroem(
        kernel="rbf",
        gamma=gamma,
        n_components=n_components,
        random_state=SEED,
    )

    classifier = LogisticRegression(
        C=C,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
        **kw,
    )

    nonlinear_model = Pipeline(
        [
            ("nystroem", rbf_features),
            ("classifier", classifier),
        ]
    )

    return _pipe(nonlinear_model)


def random_forest(
    n_estimators: int = 400,
    **kw,
):
    """Random-forest nonlinear baseline."""

    from sklearn.ensemble import RandomForestClassifier

    return _pipe(
        RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
            **kw,
        )
    )


def xgboost(
    n_estimators: int = 400,
    max_depth: int = 3,
    lr: float = 0.05,
    **kw,
):
    """Gradient-boosted-tree baseline.

    This is expected to be one of the strongest classical models on the
    70k cardiovascular dataset.
    """

    from xgboost import XGBClassifier

    return _pipe(
        XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
            **kw,
        )
    )


def mlp(
    hidden: tuple[int, ...] = (32, 16),
    **kw,
):
    """Classical neural baseline."""

    from sklearn.neural_network import MLPClassifier

    return _pipe(
        MLPClassifier(
            hidden_layer_sizes=hidden,
            max_iter=2000,
            early_stopping=True,
            random_state=SEED,
            **kw,
        )
    )


CLASSICAL = {
    "logreg": logreg,
    "svm_rbf": svm_rbf,
    "rf": random_forest,
    "xgboost": xgboost,
    "mlp": mlp,
}