"""Quantum kernel estimation: fidelity Gram matrix -> classical SVC.

Two decisions here carry most of the project's rigour, and both are about cost:

1. **The Gram matrix is computed once for the whole dataset and sliced per fold.**
   Recomputing it inside 25 folds costs ~11 hours; computing it once costs ~28
   minutes. This is only legitimate because the kernel uses no labels *and* the
   angle scaler in front of it is fixed rather than fitted -- see
   ``qheart.preprocess.FixedAngleScaler`` for the argument. Swap in a fitted scaler
   and you must set ``cache_scope="fold"``, or the benchmark is contaminated.

2. **Only the strict upper triangle is evaluated.** The fidelity kernel is symmetric
   with a unit diagonal, so ``n(n-1)/2`` evaluations suffice instead of ``n^2``.
   Free factor of two.

The kernel itself is ZZFeatureMap-style: Hadamards, then RZ angles per feature, then
ZZ interactions on each pair. The overlap ``|<phi(x_i)|phi(x_j)>|^2`` is estimated by
the adjoint trick -- run the feature map forward on x_i and its inverse on x_j, then
read the probability of the all-zeros outcome.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from qheart.quantum.budget import estimate_kernel_cost, guard_kernel_size

__all__ = ["QuantumKernel", "quantum_svc"]


def _cache_key(X: np.ndarray, n_qubits: int, reps: int, entanglement: str) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    h.update(f"|{n_qubits}|{reps}|{entanglement}".encode())
    return h.hexdigest()[:16]


class QuantumKernel:
    """Fidelity kernel over a ZZ feature map.

    ``precompute(X)`` fills the cache for every row you will ever score, then
    ``gram(idx_a, idx_b)`` returns submatrices for free. That is the only way this
    finishes in a hackathon.
    """

    def __init__(self, n_qubits: int = 4, reps: int = 2, entanglement: str = "linear",
                 shots: int | None = None, cache_dir: str | Path = "results/kernel_cache",
                 max_samples: int = 2000):
        self.n_qubits = int(n_qubits)
        self.reps = int(reps)
        self.entanglement = entanglement
        self.shots = shots
        self.cache_dir = Path(cache_dir)
        self.max_samples = int(max_samples)
        self._K: np.ndarray | None = None
        self._key: str | None = None

    # ---------------------------------------------------------------- circuit
    def _feature_map(self, x, wires) -> None:
        import pennylane as qml
        for _ in range(self.reps):
            for i, w in enumerate(wires):
                qml.Hadamard(wires=w)
                qml.RZ(2.0 * x[i], wires=w)
            pairs = ([(i, i + 1) for i in range(len(wires) - 1)]
                     if self.entanglement == "linear"
                     else [(i, j) for i in range(len(wires)) for j in range(i + 1, len(wires))])
            for i, j in pairs:
                # The (pi - x_i)(pi - x_j) product is the standard ZZFeatureMap
                # second-order term. It is what makes this map classically hard to
                # simulate -- a product of single-qubit rotations would not be.
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[j]), wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])

    def _qnode(self):
        import pennylane as qml
        dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        wires = list(range(self.n_qubits))

        @qml.qnode(dev)
        def overlap(x1, x2):
            self._feature_map(x1, wires)
            qml.adjoint(self._feature_map)(x2, wires)
            return qml.probs(wires=wires)

        return overlap

    # ---------------------------------------------------------------- gram
    def precompute(self, X: np.ndarray, verbose: bool = True) -> np.ndarray:
        """Full symmetric Gram matrix for ``X``, cached to disk by content hash."""
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_qubits:
            raise ValueError(
                f"feature map takes {self.n_qubits} features (one per qubit) but got "
                f"{X.shape[1]}. Reduce first, or raise n_qubits and pay for it."
            )
        guard_kernel_size(X.shape[0], self.max_samples)

        key = _cache_key(X, self.n_qubits, self.reps, self.entanglement)
        path = self.cache_dir / f"gram_{key}.npy"
        if path.exists():
            self._K, self._key = np.load(path), key
            if verbose:
                print(f"kernel: loaded cached Gram {self._K.shape} from {path}")
            return self._K

        est = estimate_kernel_cost(X.shape[0])
        if verbose:
            print(est.report())

        overlap = self._qnode()
        n = X.shape[0]
        K = np.eye(n)                       # unit diagonal, exactly, by construction
        for i in range(n):
            for j in range(i + 1, n):
                K[i, j] = K[j, i] = float(overlap(X[i], X[j])[0])
            if verbose and (i + 1) % 50 == 0:
                print(f"  row {i + 1}/{n}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, K)
        self._K, self._key = K, key
        return K

    def gram(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Submatrix of the precomputed Gram. Index arrays refer to ``precompute``'s X."""
        if self._K is None:
            raise RuntimeError("call precompute(X) on the full feature matrix first")
        return self._K[np.ix_(np.asarray(rows), np.asarray(cols))]

    def sanity_check(self, tol: float = 1e-6) -> None:
        """Assert the matrix is a plausible kernel. Cheap, and catches a broken
        adjoint immediately -- a non-unit diagonal or a negative eigenvalue means the
        circuit is not computing an overlap, however reasonable the accuracy looks."""
        if self._K is None:
            raise RuntimeError("nothing precomputed")
        K = self._K
        if not np.allclose(np.diag(K), 1.0, atol=max(tol, 1e-3)):
            raise AssertionError("Gram diagonal is not 1; the adjoint feature map is wrong")
        if not np.allclose(K, K.T, atol=tol):
            raise AssertionError("Gram is not symmetric")
        w = np.linalg.eigvalsh(K)
        if w.min() < -1e-6:
            raise AssertionError(
                f"Gram has a negative eigenvalue ({w.min():.2e}); with shots=None this "
                f"is a bug, with finite shots it is sampling noise -- report which."
            )


def quantum_svc(kernel: QuantumKernel, train_idx, C: float = 1.0):
    """An SVC over the precomputed kernel, exposing the harness's fit/predict_proba.

    ``probability=True`` fits Platt scaling by internal cross-validation, which is
    needed because the project reports Brier score and calibration error, and a
    decision-function value is not a probability. It also makes the fit slower and
    slightly changes the decision boundary -- worth it, but say so.
    """
    from sklearn.svm import SVC

    train_idx = np.asarray(train_idx)

    class _PrecomputedSVC:
        def __init__(self):
            self.svc = SVC(kernel="precomputed", C=C, probability=True,
                           random_state=20260830)
            self.train_idx = train_idx

        def fit(self, idx, y):
            idx = np.asarray(idx).ravel().astype(int)
            self.train_idx = idx
            self.svc.fit(kernel.gram(idx, idx), y)
            return self

        def predict_proba(self, idx):
            idx = np.asarray(idx).ravel().astype(int)
            return self.svc.predict_proba(kernel.gram(idx, self.train_idx))

    return _PrecomputedSVC()
