"""Dependency-free classifier heads, so every arm can share one byte-identical head.

Why this module exists, given that ``sklearn.linear_model.LogisticRegression`` is one
import away:

1. **The comparison is only fair if the head is identical.** This project's central
   claim is about the *feature map* -- quantum kernel, variational circuit, DCQF,
   classical control. If the quantum arm uses a PennyLane-native optimiser and the
   control uses sklearn, any difference in the results table is partly a difference in
   optimisers, and the claim collapses. One head, used by all arms, removes that
   confound entirely.
2. **sklearn is an optional dependency here** and is genuinely absent in some of the
   environments this repo has to run in. The repo's standing rule is that a missing
   quantum or ML install degrades a run, never blocks it.
3. **Parameter accounting has to be exact.** The parameter-matched control needs a
   head whose trainable-parameter count is a number this codebase can state and test,
   not one inferred from a third-party implementation's internals.

Both heads are plain maximum-likelihood logistic regression with an L2 penalty, fitted
by IRLS (Newton-Raphson). At the feature counts here -- tens, not thousands -- Newton
converges in under ten iterations and has no learning rate to tune, which matters
because an untuned learning rate is another confound between arms.

**These are not SVMs.** ``KernelLogisticHead`` takes a precomputed kernel exactly as
``sklearn.svm.SVC(kernel='precomputed')`` would, and it is a legitimate kernel method,
but it minimises log-loss and not hinge loss and it has no support-vector sparsity. A
result from it must be labelled "kernel logistic regression", never "QSVC". See
``qheart.models.kernel`` for the switch that prefers a real SVC when sklearn is
available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["LogisticHead", "KernelLogisticHead", "sigmoid"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    ``1/(1+exp(-z))`` overflows for z < -745 and silently returns 0 or nan warnings in
    the middle of a fit. The branch below evaluates the stable form on each side.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _check_binary(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).ravel()
    vals = np.unique(y)
    if not np.all(np.isin(vals, (0, 1))):
        raise ValueError(
            f"labels must be 0/1, got unique values {vals}. Encode before fitting -- "
            f"silently remapping here would hide a label-alignment bug."
        )
    if vals.size < 2:
        raise ValueError(
            "only one class present in y. This usually means a stratification bug in "
            "the fold splitter, not a modelling choice; fix the split."
        )
    return y.astype(float)


def _irls(design: np.ndarray, y: np.ndarray, penalty: np.ndarray,
          max_iter: int, tol: float) -> tuple[np.ndarray, int, bool]:
    """Newton-Raphson for penalised logistic regression.

    Solves ``(D' W D + P) delta = D'(y - p) - P w`` each step. ``P`` is a diagonal
    penalty matrix supplied by the caller so the intercept can be left unpenalised --
    penalising it shifts predictions toward 0.5 on imbalanced data, which quietly
    damages recall on the positive (diseased) class.

    A ridge is added to the Hessian when it is singular. That happens routinely here:
    DCQF outputs are strongly collinear by construction (a 3-body Z string is not
    independent of its 2-body sub-strings), and an exactly singular Hessian would
    otherwise raise mid-fold.
    """
    n_param = design.shape[1]
    w = np.zeros(n_param, dtype=float)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        p = sigmoid(design @ w)
        grad = design.T @ (y - p) - penalty * w
        s = np.clip(p * (1.0 - p), 1e-10, None)          # keep W positive definite
        hess = (design * s[:, None]).T @ design + np.diag(penalty)
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.solve(hess + 1e-8 * np.eye(n_param), grad)
        # Damped step: full Newton can overshoot into the flat tails on separable
        # data and oscillate. Halving until the log-likelihood improves is cheap.
        step = 1.0
        ll0 = _loglik(design, y, w, penalty)
        for _ in range(20):
            cand = w + step * delta
            if _loglik(design, y, cand, penalty) >= ll0:
                break
            step *= 0.5
        else:
            cand = w                                     # no improving step exists
        shift = float(np.max(np.abs(cand - w)))
        w = cand
        if shift < tol:
            converged = True
            break
    return w, it, converged


def _loglik(design: np.ndarray, y: np.ndarray, w: np.ndarray,
            penalty: np.ndarray) -> float:
    z = design @ w
    # log(1+exp(z)) computed stably as max(z,0) + log1p(exp(-|z|))
    ll = float(np.sum(y * z - (np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))))))
    return ll - 0.5 * float(np.sum(penalty * w * w))


@dataclass
class LogisticHead:
    """L2-penalised logistic regression on a feature matrix. The shared head.

    ``C`` follows sklearn's convention (inverse regularisation strength) so that a
    reader who knows sklearn is not surprised, and so the control and the quantum arms
    can be given literally the same value.
    """

    C: float = 1.0
    fit_intercept: bool = True
    max_iter: int = 100
    tol: float = 1e-8

    coef_: np.ndarray | None = field(default=None, init=False, repr=False)
    intercept_: float = field(default=0.0, init=False)
    n_iter_: int = field(default=0, init=False)
    converged_: bool = field(default=False, init=False)
    n_features_in_: int | None = field(default=None, init=False)

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ValueError("non-finite values in X; impute or clip inside the fold")
        if self.fit_intercept:
            return np.hstack([X, np.ones((X.shape[0], 1))])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticHead":
        y = _check_binary(y)
        D = self._design(X)
        if D.shape[0] != y.size:
            raise ValueError(f"X has {D.shape[0]} rows but y has {y.size}")
        self.n_features_in_ = D.shape[1] - (1 if self.fit_intercept else 0)
        if self.C <= 0:
            raise ValueError("C must be positive")
        pen = np.full(D.shape[1], 1.0 / self.C)
        if self.fit_intercept:
            pen[-1] = 0.0                                # never penalise the intercept
        w, self.n_iter_, self.converged_ = _irls(D, y, pen, self.max_iter, self.tol)
        if self.fit_intercept:
            self.coef_, self.intercept_ = w[:-1], float(w[-1])
        else:
            self.coef_, self.intercept_ = w, 0.0
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("call fit before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"predict got {X.shape[1] if X.ndim == 2 else '?'} features, fit saw "
                f"{self.n_features_in_}"
            )
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, 2) array of [P(class 0), P(class 1)], matching the harness protocol."""
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def n_trainable_params(self) -> int:
        if self.n_features_in_ is None:
            raise RuntimeError("call fit first")
        return self.n_features_in_ + (1 if self.fit_intercept else 0)


@dataclass
class KernelLogisticHead:
    """Kernel logistic regression on a **precomputed** Gram matrix.

    ``fit`` takes an (n_train, n_train) kernel; ``predict_proba`` takes an
    (n_test, n_train) cross-kernel. This is the same calling convention as
    ``sklearn.svm.SVC(kernel='precomputed')``, so the quantum kernel code can feed
    either without branching.

    Representer-theorem parameterisation: the decision function is
    ``sum_i a_i K(x, x_i) + b``, so the design matrix *is* the kernel and the
    parameter count equals the number of training samples. That has a consequence
    worth stating on a slide rather than hiding: **a kernel method's parameter count
    scales with the dataset, not with the circuit.** A 4-qubit quantum kernel fitted
    on 2000 patients carries 2001 trainable values, not 96. Quoting the circuit's
    angle count as "the model size" for a kernel method is wrong, and it is the same
    category of error this project criticises in Paper 1.
    """

    C: float = 1.0
    fit_intercept: bool = True
    max_iter: int = 100
    tol: float = 1e-8

    dual_coef_: np.ndarray | None = field(default=None, init=False, repr=False)
    intercept_: float = field(default=0.0, init=False)
    n_train_: int | None = field(default=None, init=False)
    n_iter_: int = field(default=0, init=False)
    converged_: bool = field(default=False, init=False)

    def fit(self, K: np.ndarray, y: np.ndarray) -> "KernelLogisticHead":
        K = np.asarray(K, dtype=float)
        y = _check_binary(y)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError(f"training kernel must be square, got shape {K.shape}")
        if K.shape[0] != y.size:
            raise ValueError(f"kernel is {K.shape[0]}x{K.shape[0]} but y has {y.size}")
        if not np.allclose(K, K.T, atol=1e-8):
            raise ValueError(
                "kernel is not symmetric. For a fidelity kernel this means the circuit "
                "or the caching is wrong, since |<phi(a)|phi(b)>|^2 is symmetric by "
                "construction -- do not symmetrise it away, find the bug."
            )
        self.n_train_ = K.shape[0]
        D = np.hstack([K, np.ones((K.shape[0], 1))]) if self.fit_intercept else K
        pen = np.full(D.shape[1], 1.0 / self.C)
        if self.fit_intercept:
            pen[-1] = 0.0
        w, self.n_iter_, self.converged_ = _irls(D, y, pen, self.max_iter, self.tol)
        if self.fit_intercept:
            self.dual_coef_, self.intercept_ = w[:-1], float(w[-1])
        else:
            self.dual_coef_, self.intercept_ = w, 0.0
        return self

    def decision_function(self, K: np.ndarray) -> np.ndarray:
        if self.dual_coef_ is None:
            raise RuntimeError("call fit before predicting")
        K = np.asarray(K, dtype=float)
        if K.ndim != 2 or K.shape[1] != self.n_train_:
            raise ValueError(
                f"cross-kernel must be (n_test, {self.n_train_}), got {K.shape}. A "
                f"transposed Gram block here silently scores the wrong patients."
            )
        return K @ self.dual_coef_ + self.intercept_

    def predict_proba(self, K: np.ndarray) -> np.ndarray:
        p1 = sigmoid(self.decision_function(K))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, K: np.ndarray) -> np.ndarray:
        return (self.predict_proba(K)[:, 1] >= 0.5).astype(int)

    def n_trainable_params(self) -> int:
        if self.n_train_ is None:
            raise RuntimeError("call fit first")
        return self.n_train_ + (1 if self.fit_intercept else 0)
