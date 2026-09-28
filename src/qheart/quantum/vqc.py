"""Variational quantum classifier. 4 qubits, dense angle encoding, depth 6, 96 angles.

Trained with PennyLane's autograd interface on ``default.qubit``, which
backpropagates through the simulator and is far faster than parameter-shift. That
raises an honest question for the deck, so here is the answer to give: backprop
through a statevector simulator is not physically available on hardware. On hardware
each gradient component needs the **parameter-shift rule**: two circuit evaluations
per parameter per step, so 96 parameters is 192 evaluations per step, times batches,
times epochs. ``gradient_method="parameter-shift"`` reproduces the hardware-honest
cost and should be run once, briefly, so the reported training cost is measured
rather than asserted.
"""

from __future__ import annotations

import numpy as np

from qheart.quantum.ansatz import assert_param_count, init_weights, layered_ansatz
from qheart.quantum.encodings import dense_angle_encode, n_qubits_for

__all__ = ["VQC"]


class VQC:
    """Satisfies the harness's ``fit`` / ``predict_proba`` contract.

    ``predict_proba`` maps the Pauli-Z expectation on wire 0, which lives in
    [-1, 1], to [0, 1] by ``(1 + z) / 2``. That is a linear rescaling, not a
    calibrated probability -- Brier score and ECE on a raw VQC output are usually
    poor for that reason alone. Fit a sigmoid on the training fold if you want
    calibrated numbers, and report that you did.
    """

    def __init__(self, n_features: int = 8, depth: int = 6, epochs: int = 60,
                 lr: float = 0.05, batch_size: int = 32, seed: int = 20260830,
                 entangler: str = "ring", gradient_method: str = "backprop",
                 verbose: bool = False):
        self.n_features = int(n_features)
        self.n_qubits = n_qubits_for(n_features, "dense_angle")
        self.depth = int(depth)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.entangler = entangler
        self.gradient_method = gradient_method
        self.verbose = verbose
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.n_trainable_params: int | None = None
        self.loss_curve_: list[float] = []
        self.circuit_evaluations_: int = 0

    # ---------------------------------------------------------------- circuit
    def _build(self):
        import pennylane as qml
        diff = "backprop" if self.gradient_method == "backprop" else "parameter-shift"
        dev = qml.device("default.qubit", wires=self.n_qubits)
        wires = list(range(self.n_qubits))

        @qml.qnode(dev, interface="autograd", diff_method=diff)
        def circuit(x, weights):
            dense_angle_encode(x, wires)
            layered_ansatz(weights, wires, entangler=self.entangler)
            return qml.expval(qml.PauliZ(0))

        return circuit

    # ---------------------------------------------------------------- fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQC":
        import pennylane as qml
        from pennylane import numpy as pnp

        from qheart.seeds import spawn

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if X.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {X.shape[1]}")

        circuit = self._build()
        w = pnp.array(init_weights(self.n_qubits, self.depth, self.seed), requires_grad=True)
        b = pnp.array(0.0, requires_grad=True)
        self.n_trainable_params = assert_param_count(w, self.n_qubits, self.depth) + 1

        # Class weights, because a threshold alone cannot fix a loss that was never
        # told the classes matter differently. On roughly balanced data these are
        # near 1.0 and harmless; on skewed data they are what keeps sensitivity up.
        pos = float((y == 1).mean())
        cw = {0: 0.5 / max(1e-9, 1 - pos), 1: 0.5 / max(1e-9, pos)}
        t = 2.0 * y - 1.0                    # {0,1} -> {-1,+1} to match PauliZ's range
        sw = np.array([cw[int(v)] for v in y])

        def cost(w, b, xb, tb, sb):
            preds = pnp.stack([circuit(xi, w) for xi in xb]) + b
            return pnp.mean(sb * (preds - tb) ** 2)

        opt = qml.AdamOptimizer(stepsize=self.lr)
        rng = spawn(self.seed, "vqc_batches")
        n = X.shape[0]
        evals_per_step = 0
        for ep in range(self.epochs):
            order = rng.permutation(n)
            ep_loss = 0.0
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                xb = pnp.array(X[idx], requires_grad=False)
                tb = pnp.array(t[idx], requires_grad=False)
                sb = pnp.array(sw[idx], requires_grad=False)
                (w, b, _, _, _), loss = opt.step_and_cost(cost, w, b, xb, tb, sb)
                ep_loss += float(loss) * len(idx)
                # Hardware-honest accounting: 2 evaluations per parameter per sample
                # under parameter-shift, 1 forward pass per sample under backprop.
                evals_per_step = (2 * self.n_trainable_params * len(idx)
                                  if self.gradient_method != "backprop" else len(idx))
                self.circuit_evaluations_ += evals_per_step
            self.loss_curve_.append(ep_loss / n)
            if self.verbose and (ep + 1) % 10 == 0:
                print(f"    epoch {ep + 1}/{self.epochs} loss={self.loss_curve_[-1]:.4f}")

        self.weights_, self.bias_, self._circuit = np.array(w), float(b), circuit
        self._flat_loss_warning()
        return self

    def _flat_loss_warning(self) -> None:
        """Detect a barren plateau instead of reporting chance accuracy as a result."""
        c = self.loss_curve_
        if len(c) >= 10 and abs(c[0] - c[-1]) < 1e-3:
            print("WARNING: VQC loss barely moved across training. This is the barren "
                  "plateau signature, not a hard problem. Reduce depth, shrink the "
                  "init scale in ansatz.init_weights, or use fewer qubits. Do not "
                  "report the resulting accuracy as a quantum result.")

    # ---------------------------------------------------------------- predict
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("fit first")
        X = np.asarray(X, dtype=float)
        return np.array([float(self._circuit(xi, self.weights_)) + self.bias_ for xi in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.clip(self.decision_function(X), -1.0, 1.0)
        p = (1.0 + z) / 2.0
        return np.column_stack([1.0 - p, p])
