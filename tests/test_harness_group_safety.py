import numpy as np

from qheart.eval.harness import evaluate


class DummyModel:
    def fit(self, X, y):
        self.p = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.column_stack([1.0 - p, p])


def test_evaluate_accepts_groups():
    rng = np.random.default_rng(20260830)

    y = np.array(
        [0, 1] * 50,
        dtype=int,
    )

    X = rng.normal(
        size=(100, 2),
    )

    # Each pair belongs to one group.
    groups = np.repeat(
        np.arange(50),
        2,
    )

    result = evaluate(
        lambda: DummyModel(),
        X,
        y,
        model_name="dummy_group_test",
        groups=groups,
        n_splits=5,
        n_repeats=2,
        seed=20260830,
        tune_threshold=False,
        verbose=False,
    )

    assert len(result.per_fold) == 10
    assert result.oof_proba is not None
    assert np.isfinite(result.oof_proba).all()