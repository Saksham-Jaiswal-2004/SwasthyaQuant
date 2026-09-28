"""Quantum feature extractor plus classical head -- the HQF-CC pattern from P1.

The circuit outputs one expectation value per qubit; those n numbers are the feature
vector, and a small classical classifier sits on top. This is the route that makes
the parameter-efficiency claim legible, because the head can be held byte-identical
between the quantum extractor and Control-C. If the two heads differ, the comparison
measures the heads.

The default head is logistic regression, not an MLP. A logistic head has no hidden
capacity of its own, so whatever separates the classes was produced by the layer in
front of it -- which is the thing under test.
"""

from __future__ import annotations

import numpy as np

from qheart.quantum.ansatz import assert_param_count, init_weights, layered_ansatz
from qheart.quantum.encodings import dense_angle_encode, n_qubits_for

__all__ = ["QuantumExtractor", "HybridExtractorClassifier"]


class QuantumExtractor:
    """Transform-only: features in, ``n_qubits`` expectation values out.

    ``trainable=False`` by default, i.e. the extractor is a **fixed random circuit**.
    That sounds like a weakness and is actually the cleanest version of the
    experiment: a fixed random feature map with 96 angles versus a classical random
    projection with ~96 parameters isolates the effect of the quantum feature space
    from the effect of training. Train it end-to-end afterwards, as a second
    condition, and report both.
    """

    def __init__(self, n_features: int = 8, depth: int = 6, seed: int = 20260830,
                 entangler: str = "ring", trainable: bool = False):
        self.n_features = int(n_features)
        self.n_qubits = n_qubits_for(n_features, "dense_angle")
        self.depth = int(depth)
        self.seed = int(seed)
        self.entangler = entangler
        self.trainable = bool(trainable)
        self.weights_ = init_weights(self.n_qubits, self.depth, self.seed)
        self.n_trainable_params = (assert_param_count(self.weights_, self.n_qubits, self.depth)
                                   if trainable else 0)
        self.n_circuit_angles = int(self.weights_.size)
        self._circuit = None

    def _build(self):
        import pennylane as qml
        dev = qml.device("default.qubit", wires=self.n_qubits)
        wires = list(range(self.n_qubits))

        @qml.qnode(dev, interface="autograd", diff_method="backprop")
        def circuit(x, weights):
            dense_angle_encode(x, wires)
            layered_ansatz(weights, wires, entangler=self.entangler)
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        return circuit

    def fit(self, X=None, y=None) -> "QuantumExtractor":
        """No-op unless ``trainable``; the circuit is fixed by its seed.

        Present so the object drops into a scikit-learn Pipeline unchanged, which is
        how it gets fitted inside the fold along with the scaler.
        """
        if self._circuit is None:
            self._circuit = self._build()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._circuit is None:
            self.fit()
        X = np.asarray(X, dtype=float)
        return np.array([np.asarray(self._circuit(xi, self.weights_), dtype=float).ravel()
                         for xi in X])

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class HybridExtractorClassifier:
    """Quantum extractor -> classical head, satisfying the harness contract.

    The head is constructed by ``head_factory`` so that Control-C can be handed the
    *same* factory. Same head, same hyperparameters, same random state; only the
    layer in front differs. That is the whole design.
    """

    def __init__(self, extractor: QuantumExtractor, head_factory=None):
        self.extractor = extractor
        self.head_factory = head_factory or _default_head
        self.head_ = None
        self.n_trainable_params: int | None = None

    def fit(self, X, y):
        Z = self.extractor.fit_transform(X, y)
        self.head_ = self.head_factory()
        self.head_.fit(Z, y)
        head_params = int(np.asarray(self.head_.coef_).size + np.asarray(self.head_.intercept_).size)
        self.n_trainable_params = self.extractor.n_trainable_params + head_params
        self.n_circuit_angles = self.extractor.n_circuit_angles
        self.n_head_params = head_params
        return self

    def predict_proba(self, X):
        if self.head_ is None:
            raise RuntimeError("fit first")
        return self.head_.predict_proba(self.extractor.transform(X))


def _default_head():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, random_state=20260830)
