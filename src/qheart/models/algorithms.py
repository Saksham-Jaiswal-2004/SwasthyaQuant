"""QSVC, VQC and Hybrid QNN -- three quantum classifiers on one identical pipeline.

The point of putting all three in one module is that they share everything except the
part being compared. Same encoding, same qubit count, same statevector backend, same
``LogisticHead`` where a head is needed, same ``fit``/``predict_proba`` protocol from
``qheart.eval.harness``. When the results table shows a difference between these rows,
the difference is the algorithm and nothing else.

What each one actually is:

``QSVC``      Quantum kernel method. Encode each patient into a state; the kernel is
              the fidelity |<phi(a)|phi(b)>|^2 between two patients' states. All the
              "quantumness" is in the similarity measure; the classifier on top is a
              standard convex solver. **Zero trainable quantum parameters** -- but see
              the parameter-count warning below, because a kernel method's parameter
              count is not zero, it is O(N).

``VQC``       Variational classifier. Encode, then a trainable circuit, then read one
              Pauli-Z expectation as the decision value. Trained end-to-end by
              parameter-shift gradients. This is the only one of the three where the
              quantum circuit itself is optimised against the loss, and therefore the
              only one where "96 trainable angles = 384 bytes" describes the model.

``HybridQNN`` Quantum feature extractor plus a classical head. Encode, run the circuit,
              measure several Z-string expectations, feed those to logistic regression.
              With ``trainable=False`` (the shipped default, ADR-008) the circuit is a
              fixed random map with zero trainable parameters and the entire trainable
              budget is the head. This is P1's architecture and the arm the
              parameter-matched control was built for.

**The parameter-count trap, stated once, loudly.** These three do not have comparable
parameter counts and it is wrong to put them in a table under one "params" column
without a footnote:

    VQC        96 angles + 1 bias        = 97 values,  388 bytes
    HybridQNN  0 quantum (fixed) + head  = head only, and the head's size depends
                                           on how many observables you measure
    QSVC       0 quantum + N+1 dual coefficients, so it GROWS WITH THE DATASET --
               2001 values on a 2000-patient subsample, not 96

Quoting the circuit's angle count as a QSVC's model size is the same error this
project criticises Paper 1 for. ``n_trainable_params()`` on each class returns the
honest number, and ``parameter_report()`` prints all three side by side with the
caveat attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qheart.models.heads import KernelLogisticHead, LogisticHead
from qheart.quantum.ansatz import init_weights, weight_shape
from qheart.quantum.circuits import (apply_ansatz, encode, gram_matrix, n_weights,
                                     prepare)
from qheart.quantum.encodings import n_qubits_for
from qheart.quantum.statevector import State

__all__ = ["QSVC", "VQC", "HybridQNN", "parameter_report"]

_FLOAT32_BYTES = 4


def _try_sklearn_svc(C: float):
    """Return a real precomputed-kernel SVC if sklearn is installed, else None.

    Split out so the import failure is handled in exactly one place and so the
    fallback is a deliberate, labelled substitution rather than a silent one.
    """
    try:
        from sklearn.svm import SVC
    except Exception:
        return None
    return SVC(C=C, kernel="precomputed", probability=False)


# ------------------------------------------------------------------------ QSVC
@dataclass
class QSVC:
    """Fidelity-kernel classifier.

    ``head_kind`` reports what actually ran. With sklearn present this is a true
    support vector classifier and the name QSVC is accurate. Without it, the fallback
    is kernel logistic regression -- a legitimate kernel method, but it minimises
    log-loss rather than hinge loss and has no support-vector sparsity, so a result
    from it must be labelled "quantum kernel + logistic", never "QSVC". The attribute
    exists so a results table can carry the right label automatically instead of
    relying on whoever writes the slide to remember.
    """

    n_qubits: int = 4
    encoding: str = "dense_angle"
    C: float = 1.0
    prefer_sklearn: bool = True

    head_: object = field(default=None, init=False, repr=False)
    head_kind_: str = field(default="", init=False)
    X_train_: np.ndarray | None = field(default=None, init=False, repr=False)
    n_features_in_: int | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QSVC":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError(f"expected 2-D X, got {X.shape}")
        self.n_features_in_ = X.shape[1]
        self.X_train_ = X.copy()          # a kernel method must keep its training set
        K = gram_matrix(X, None, self.n_qubits, self.encoding)

        svc = _try_sklearn_svc(self.C) if self.prefer_sklearn else None
        if svc is not None:
            self.head_, self.head_kind_ = svc.fit(K, y), "sklearn.svm.SVC(precomputed)"
        else:
            self.head_ = KernelLogisticHead(C=self.C).fit(K, y)
            self.head_kind_ = "KernelLogisticHead (sklearn absent; NOT an SVC)"
        return self

    def _cross_kernel(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("call fit before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"predict got {X.shape} , fit saw {self.n_features_in_} features")
        return gram_matrix(X, self.X_train_, self.n_qubits, self.encoding, symmetric=False)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        K = self._cross_kernel(X)
        if hasattr(self.head_, "predict_proba") and isinstance(self.head_, KernelLogisticHead):
            return self.head_.predict_proba(K)
        # sklearn SVC without probability=True: map the margin through a logistic.
        # This is a monotone transform, so AUC is exact and thresholded metrics at 0.5
        # are NOT calibrated. Report AUC from this, not Brier score.
        d = np.asarray(self.head_.decision_function(K), dtype=float).ravel()
        p1 = 1.0 / (1.0 + np.exp(-d))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def n_trainable_params(self) -> int:
        """N+1, not zero and not 96. See the module docstring."""
        if self.X_train_ is None:
            raise RuntimeError("call fit first")
        if isinstance(self.head_, KernelLogisticHead):
            return self.head_.n_trainable_params()
        return int(self.X_train_.shape[0]) + 1

    def n_quantum_params(self) -> int:
        return 0


# ------------------------------------------------------------------------- VQC
@dataclass
class VQC:
    """Variational quantum classifier, trained by parameter-shift gradients.

    Decision value is <Z_0> after encoding and ansatz, scaled and shifted by two
    trainable classical scalars, then squashed through a logistic. The scale matters:
    <Z_0> lives in [-1, 1], so without it the model cannot express a confident
    probability and the loss plateaus in a way that looks like barren-plateau
    behaviour but is only a missing output scale.

    **Cost.** Parameter-shift needs two circuit evaluations per parameter per sample,
    so one gradient step on n samples with p parameters is 2*n*p circuits. At 400
    samples and 96 parameters that is 76,800 circuits per step. This is exact and
    hardware-faithful, and it is why the default epoch count is small and why the VQC
    arm is the slowest thing in this repo. ``finite_diff=True`` swaps in central
    differences for a quick smoke test; it is not for reported results.
    """

    n_qubits: int = 4
    depth: int = 6
    encoding: str = "dense_angle"
    entangler: str = "ring"
    epochs: int = 30
    lr: float = 0.15
    batch_size: int = 32
    l2: float = 1e-4
    seed: int = 20260830
    finite_diff: bool = False
    verbose: bool = False

    weights_: np.ndarray | None = field(default=None, init=False, repr=False)
    scale_: float = field(default=2.0, init=False)
    bias_: float = field(default=0.0, init=False)
    loss_: list[float] = field(default_factory=list, init=False, repr=False)
    n_features_in_: int | None = field(default=None, init=False)

    # -------------------------------------------------------------- forward
    def _z0(self, x: np.ndarray, w: np.ndarray) -> float:
        st = State(self.n_qubits)
        encode(st, x, self.encoding)
        apply_ansatz(st, w, self.entangler)
        return st.z_expectation([0])

    def _decision(self, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        z = np.array([self._z0(r, w) for r in X], dtype=float)
        return self.scale_ * z + self.bias_

    # -------------------------------------------------------------- gradients
    def _grad_z0(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """d<Z_0>/d(theta) for every angle.

        Parameter-shift: for a gate exp(-i theta P / 2) with P^2 = I,
        df/dtheta = [f(theta + pi/2) - f(theta - pi/2)] / 2. Exact, not an
        approximation -- which is the whole reason to prefer it to finite differences,
        since it survives shot noise and has no step size to tune.
        """
        flat = w.ravel()
        g = np.empty(flat.size, dtype=float)
        if self.finite_diff:
            h = 1e-5
            for i in range(flat.size):
                up, dn = flat.copy(), flat.copy()
                up[i] += h; dn[i] -= h
                g[i] = (self._z0(x, up.reshape(w.shape)) - self._z0(x, dn.reshape(w.shape))) / (2 * h)
            return g.reshape(w.shape)
        for i in range(flat.size):
            up, dn = flat.copy(), flat.copy()
            up[i] += np.pi / 2
            dn[i] -= np.pi / 2
            g[i] = 0.5 * (self._z0(x, up.reshape(w.shape))
                          - self._z0(x, dn.reshape(w.shape)))
        return g.reshape(w.shape)

    # -------------------------------------------------------------- sklearn API
    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQC":
        from qheart.models.heads import sigmoid

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel().astype(float)
        if X.ndim != 2:
            raise ValueError(f"expected 2-D X, got {X.shape}")
        if not np.all(np.isin(np.unique(y), (0.0, 1.0))):
            raise ValueError("VQC needs 0/1 labels")
        self.n_features_in_ = X.shape[1]
        cap = 2 * self.n_qubits if self.encoding == "dense_angle" else self.n_qubits
        if X.shape[1] > cap:
            raise ValueError(
                f"{X.shape[1]} features exceed the {cap} that {self.encoding} holds on "
                f"{self.n_qubits} qubits. Reduce features or add qubits -- do not let "
                f"the encoder drop columns silently."
            )

        rng = np.random.default_rng(self.seed)
        w = init_weights(self.n_qubits, self.depth, seed=self.seed)
        w = np.asarray(w, dtype=float).reshape(weight_shape(self.n_qubits, self.depth))
        self.scale_, self.bias_ = 2.0, 0.0
        n = X.shape[0]
        self.loss_ = []

        for ep in range(self.epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                Xb, yb = X[idx], y[idx]
                zb = np.array([self._z0(r, w) for r in Xb])
                pb = sigmoid(self.scale_ * zb + self.bias_)
                err = pb - yb                                  # d(logloss)/d(decision)
                # chain rule: d/dtheta = sum_b err_b * scale * dz_b/dtheta
                gw = np.zeros_like(w)
                for b, r in enumerate(Xb):
                    if abs(err[b]) > 1e-12:
                        gw += err[b] * self.scale_ * self._grad_z0(r, w)
                gw = gw / len(idx) + self.l2 * w
                gs = float(np.mean(err * zb))
                gbias = float(np.mean(err))
                w -= self.lr * gw
                self.scale_ -= self.lr * gs
                self.bias_ -= self.lr * gbias
            zt = np.array([self._z0(r, w) for r in X])
            pt = np.clip(sigmoid(self.scale_ * zt + self.bias_), 1e-12, 1 - 1e-12)
            ll = float(-np.mean(y * np.log(pt) + (1 - y) * np.log(1 - pt)))
            self.loss_.append(ll)
            if self.verbose:
                print(f"  epoch {ep+1:>3}  logloss {ll:.5f}")
        self.weights_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        from qheart.models.heads import sigmoid
        if self.weights_ is None:
            raise RuntimeError("call fit before predicting")
        X = np.asarray(X, dtype=float)
        p1 = sigmoid(self._decision(X, self.weights_))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def n_quantum_params(self) -> int:
        return n_weights(self.n_qubits, self.depth)

    def n_trainable_params(self) -> int:
        """Angles plus the output scale and bias. The two classical scalars are
        trainable and are part of the model; omitting them to reach a rounder number
        is exactly the kind of shaving this project refuses to do."""
        return self.n_quantum_params() + 2

    def payload_bytes(self, dtype_bytes: int = _FLOAT32_BYTES) -> int:
        return self.n_trainable_params() * dtype_bytes


# ------------------------------------------------------------------ HybridQNN
@dataclass
class HybridQNN:
    """Quantum feature extractor + classical head. P1's architecture.

    ``trainable=False`` is the default (ADR-008): the circuit weights are fixed at
    their random initialisation and never optimised, so the quantum stage contributes
    **zero** trainable parameters and the arm is matched by a zero-parameter classical
    reducer. That is the cleanest form of this project's claim, and it is also the
    honest reading of what a frozen extractor is.

    ``trainable=True`` optimises the circuit through the head by parameter-shift, which
    costs the same 2*n*p circuits per step as the VQC. Say which one was run; the two
    have different parameter counts and different stories.
    """

    n_qubits: int = 4
    depth: int = 6
    encoding: str = "dense_angle"
    entangler: str = "ring"
    observables: tuple[tuple[int, ...], ...] | None = None
    trainable: bool = False
    C: float = 1.0
    seed: int = 20260830

    weights_: np.ndarray | None = field(default=None, init=False, repr=False)
    head_: LogisticHead | None = field(default=None, init=False, repr=False)
    observables_: list[tuple[int, ...]] = field(default_factory=list, init=False, repr=False)
    n_features_in_: int | None = field(default=None, init=False)

    def _default_observables(self) -> list[tuple[int, ...]]:
        singles = [(i,) for i in range(self.n_qubits)]
        pairs = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)] \
            if self.n_qubits > 2 else [(0, 1)]
        return singles + pairs

    def _features(self, X: np.ndarray) -> np.ndarray:
        out = np.empty((X.shape[0], len(self.observables_)), dtype=float)
        for i, r in enumerate(X):
            st = prepare(r, self.n_qubits, self.encoding, self.weights_, self.entangler)
            out[i] = st.z_string_expectations(self.observables_)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridQNN":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected 2-D X, got {X.shape}")
        self.n_features_in_ = X.shape[1]
        self.observables_ = list(self.observables) if self.observables \
            else self._default_observables()
        w = init_weights(self.n_qubits, self.depth, seed=self.seed)
        self.weights_ = np.asarray(w, dtype=float).reshape(
            weight_shape(self.n_qubits, self.depth))
        if self.trainable:
            raise NotImplementedError(
                "trainable=True needs backpropagation through the head into the "
                "circuit. It is a real code path, not a config flag -- implement it "
                "deliberately rather than by loosening this check. The frozen "
                "extractor (ADR-008) is the shipped default and the one the "
                "parameter-matched control is sized against."
            )
        self.head_ = LogisticHead(C=self.C).fit(self._features(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.head_ is None:
            raise RuntimeError("call fit before predicting")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"predict got {X.shape}, fit saw {self.n_features_in_} features")
        return self.head_.predict_proba(self._features(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def n_quantum_params(self) -> int:
        return n_weights(self.n_qubits, self.depth) if self.trainable else 0

    def n_trainable_params(self) -> int:
        if self.head_ is None:
            raise RuntimeError("call fit first")
        return self.n_quantum_params() + self.head_.n_trainable_params()


# ------------------------------------------------------------------ reporting
def parameter_report(models: dict[str, object]) -> str:
    """Side-by-side parameter counts with the comparability caveat attached.

    The caveat is not optional garnish. A table with one "params" column across a
    kernel method, a variational circuit and a frozen extractor invites exactly the
    apples-to-oranges comparison this project exists to avoid, so the warning ships
    with the numbers rather than being left to a footnote someone may drop.
    """
    lines = [f"{'model':<14}{'quantum':>10}{'trainable':>11}{'bytes@fp32':>12}"]
    lines.append("-" * 47)
    for name, m in models.items():
        q = m.n_quantum_params() if hasattr(m, "n_quantum_params") else 0
        try:
            t = m.n_trainable_params()
        except RuntimeError:
            t = -1
        lines.append(f"{name:<14}{q:>10}{t:>11}{t * _FLOAT32_BYTES:>12}")
    lines.append("")
    lines.append("CAVEAT: these counts are NOT comparable across rows.")
    lines.append("  QSVC      grows with the training set (N+1 dual coefficients).")
    lines.append("  VQC       is angles + output scale + bias; this one number is the model.")
    lines.append("  HybridQNN with trainable=False has zero quantum parameters by design;")
    lines.append("            its count is the classical head alone.")
    lines.append("Quote a circuit's angle count as 'model size' only for the VQC row.")
    return "\n".join(lines)
