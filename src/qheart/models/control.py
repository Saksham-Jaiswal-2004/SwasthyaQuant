"""Control-C -- the parameter-matched classical control. The project's contribution.

Neither reference paper has one. Without it, "the quantum model reaches 0.89 AUC
with 96 parameters" is unfalsifiable: nobody knows what a classical model with 96
parameters would have reached, so the sentence carries no information. Control-C is
the sentence that makes it mean something.

Two controls, matched to the two quantum conditions:

* ``FixedRandomReducer`` pairs with ``QuantumExtractor(trainable=False)``. Both are
  *untrained random feature maps* with the same parameter count, feeding an identical
  head. This is the cleanest experiment available here, because training is held
  constant at zero on both sides and only the nature of the feature space differs.
* ``TrainableControlReducer`` pairs with the trained VQC or extractor. Torch, so
  gradients are available, still parameter-matched.

The head is supplied by the *same* factory the quantum path uses. If the heads
differ, the experiment measures the heads.
"""

from __future__ import annotations

import numpy as np

from qheart.params import assert_matched, size_control_reducer

__all__ = ["FixedRandomReducer", "TrainableControlReducer", "ControlClassifier",
           "build_control_for"]


class FixedRandomReducer:
    """``d_in -> h -> d_out`` with tanh, weights drawn once and never trained.

    tanh, not ReLU, because the quantum extractor emits Pauli-Z expectations bounded
    in [-1, 1]. Matching the output range means the shared head sees inputs on the
    same scale from both sides; a ReLU control would hand the head unbounded
    non-negative features and the comparison would partly be about scaling.
    """

    def __init__(self, d_in: int, d_out: int = 4, target_params: int = 96,
                 hidden: int | None = None, seed: int = 20260830, scale: float = 1.0):
        sizing = size_control_reducer(target_params, d_in, d_out)
        self.sizing = sizing
        self.hidden = int(hidden if hidden is not None else sizing.hidden_below)
        self.d_in, self.d_out = int(d_in), int(d_out)
        self.seed = int(seed)

        from qheart.seeds import spawn
        rng = spawn(seed, "control_fixed_random")
        # 1/sqrt(fan_in) keeps pre-activations near unit variance, so tanh operates in
        # its responsive region instead of saturating -- a saturated control is a
        # handicapped control, and a handicapped control proves nothing.
        self.W1 = rng.normal(0, scale / np.sqrt(d_in), size=(d_in, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, scale / np.sqrt(self.hidden), size=(self.hidden, d_out))
        self.b2 = np.zeros(d_out)
        self.n_trainable_params = 0
        self.n_parameters = int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def fit(self, X=None, y=None) -> "FixedRandomReducer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.tanh(np.tanh(X @ self.W1 + self.b1) @ self.W2 + self.b2)

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.transform(X)


class TrainableControlReducer:
    """Torch ``d_in -> h -> d_out`` with tanh, trained by gradient descent.

    Matched to the trained quantum condition. Torch is imported lazily so the
    classical-only install is unaffected.
    """

    def __init__(self, d_in: int, d_out: int = 4, target_params: int = 96,
                 hidden: int | None = None, epochs: int = 200, lr: float = 0.01,
                 seed: int = 20260830):
        self.sizing = size_control_reducer(target_params, d_in, d_out)
        self.hidden = int(hidden if hidden is not None else self.sizing.hidden_below)
        self.d_in, self.d_out = int(d_in), int(d_out)
        self.epochs, self.lr, self.seed = int(epochs), float(lr), int(seed)
        self.net_ = None
        self.n_trainable_params: int | None = None
        self.loss_curve_: list[float] = []

    def _build(self):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        return nn.Sequential(nn.Linear(self.d_in, self.hidden), nn.Tanh(),
                             nn.Linear(self.hidden, self.d_out), nn.Tanh())

    def fit(self, X, y) -> "TrainableControlReducer":
        import torch
        import torch.nn as nn

        from qheart.params import count_torch_params

        self.net_ = self._build()
        self.n_trainable_params = count_torch_params(self.net_)
        # A linear read-out only for training the reducer; it is discarded afterwards
        # so the shared head remains the only classifier in the comparison.
        probe = nn.Linear(self.d_out, 1)
        opt = torch.optim.Adam(list(self.net_.parameters()) + list(probe.parameters()),
                              lr=self.lr)
        Xt = torch.tensor(np.asarray(X, dtype=np.float32))
        yt = torch.tensor(np.asarray(y, dtype=np.float32)).view(-1, 1)
        pos = float(yt.mean())
        w = torch.where(yt > 0.5, 0.5 / max(1e-9, pos), 0.5 / max(1e-9, 1 - pos))
        lossf = nn.BCEWithLogitsLoss(reduction="none")
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = (lossf(probe(self.net_(Xt)), yt) * w).mean()
            loss.backward()
            opt.step()
            self.loss_curve_.append(float(loss))
        return self

    def transform(self, X):
        import torch
        if self.net_ is None:
            raise RuntimeError("fit first")
        with torch.no_grad():
            return self.net_(torch.tensor(np.asarray(X, dtype=np.float32))).numpy()

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class ControlClassifier:
    """Reducer -> shared head. Mirrors ``HybridExtractorClassifier`` exactly.

    ``assert_matched`` runs on every fit. A control that has silently drifted out of
    match after an unrelated config change is worse than no control, because it is
    still labelled the control.
    """

    def __init__(self, reducer, head_factory=None, quantum_param_count: int | None = None,
                 tolerance: float = 0.15):
        self.reducer = reducer
        self.head_factory = head_factory or _default_head
        self.quantum_param_count = quantum_param_count
        self.tolerance = tolerance
        self.head_ = None
        self.n_trainable_params: int | None = None

    def fit(self, X, y):
        Z = self.reducer.fit_transform(X, y)
        self.head_ = self.head_factory()
        self.head_.fit(Z, y)
        head_params = int(np.asarray(self.head_.coef_).size
                          + np.asarray(self.head_.intercept_).size)
        reducer_params = int(getattr(self.reducer, "n_parameters", None)
                             or getattr(self.reducer, "n_trainable_params", 0))
        self.n_reducer_params = reducer_params
        self.n_head_params = head_params
        self.n_trainable_params = reducer_params + head_params
        if self.quantum_param_count:
            assert_matched(self.quantum_param_count, reducer_params, self.tolerance)
        return self

    def predict_proba(self, X):
        if self.head_ is None:
            raise RuntimeError("fit first")
        return self.head_.predict_proba(self.reducer.transform(X))


def _default_head():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, random_state=20260830)


def build_control_for(quantum_model, d_in: int, trainable: bool | None = None,
                      head_factory=None, seed: int = 20260830) -> ControlClassifier:
    """Construct the matched control **from the quantum model itself**, at runtime.

    This is the guard against the most likely quiet failure in the project: someone
    changes the circuit depth, the control stays at its hard-coded width, and the
    headline comparison silently becomes unmatched while every number still looks
    reasonable. Reading the target off the live object makes that impossible.
    """
    target = (getattr(quantum_model, "n_circuit_angles", None)
              or getattr(quantum_model, "n_trainable_params", None))
    if not target:
        raise ValueError(
            "cannot read a parameter count off the quantum model; expose "
            "n_circuit_angles or n_trainable_params before building its control"
        )
    d_out = getattr(getattr(quantum_model, "extractor", quantum_model), "n_qubits", 4)
    train = (getattr(getattr(quantum_model, "extractor", quantum_model), "trainable", True)
             if trainable is None else trainable)
    cls = TrainableControlReducer if train else FixedRandomReducer
    reducer = cls(d_in=d_in, d_out=d_out, target_params=int(target), seed=seed)
    return ControlClassifier(reducer, head_factory=head_factory,
                             quantum_param_count=int(target))
