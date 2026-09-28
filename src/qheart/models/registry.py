"""One name -> one model. The registry is what keeps the comparison fair.

Every model in the project, classical or quantum, is built through ``build(name)``
and scored through ``qheart.eval.harness.evaluate``. There is no second code path, so
there is no way for the quantum models to be evaluated under quieter rules than the
baselines -- which, more than any single metric, is what makes a benchmark table
worth reading.

``ModelSpec.input_kind`` exists because the quantum kernel is the one model that does
not take a feature matrix: an SVC over a precomputed Gram matrix takes *row indices*
into the cached kernel. Declaring that in the registry keeps the special case in one
readable place instead of scattering ``if name == "qkernel"`` through the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["ModelSpec", "REGISTRY", "available", "build", "spec", "describe",
           "quantum_models", "classical_models"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[..., Any]
    needs_quantum: bool = False
    input_kind: str = "features"          # "features" | "kernel_indices"
    reports_params: bool = False
    note: str = ""
    defaults: dict = field(default_factory=dict)


def _q(name):
    """Import a quantum model lazily so ``available()`` works without pennylane."""
    def factory(**kw):
        if name == "vqc":
            from qheart.quantum.vqc import VQC
            return VQC(**kw)
        if name == "hybrid_extractor":
            from qheart.quantum.extractor import (HybridExtractorClassifier,
                                                  QuantumExtractor)
            ex_kw = {k: kw.pop(k) for k in list(kw)
                     if k in {"n_features", "depth", "seed", "entangler", "trainable"}}
            return HybridExtractorClassifier(QuantumExtractor(**ex_kw), **kw)
        raise KeyError(name)
    return factory


def _control(**kw):
    from qheart.models.control import (ControlClassifier, FixedRandomReducer,
                                       TrainableControlReducer)
    trainable = kw.pop("trainable", False)
    target = kw.pop("target_params", 96)
    d_in = kw.pop("d_in", 8)
    d_out = kw.pop("d_out", 4)
    cls = TrainableControlReducer if trainable else FixedRandomReducer
    return ControlClassifier(cls(d_in=d_in, d_out=d_out, target_params=target, **kw),
                            quantum_param_count=target)


def _qkernel(**kw):
    from qheart.quantum.kernel import QuantumKernel
    return QuantumKernel(**kw)


def _backend(name):
    """NumPy-backend models. ``needs_quantum=False`` because they genuinely do not
    need one -- they run on ``qheart.quantum.statevector``, which is numpy only.

    Registering them as classical-availability is not a fudge: the point is that the
    quantum *circuits* are executable in any environment, so a broken pennylane build
    degrades hardware fidelity, not the ability to reproduce the table.
    """
    def factory(**kw):
        from qheart.models import algorithms as A
        from qheart.models.dcqf_control import ChainProductControl
        from qheart.quantum.dcqf import DCQFExtractor
        if name == "np_qsvc":
            return A.QSVC(**kw)
        if name == "np_vqc":
            return A.VQC(**kw)
        if name == "np_hybrid":
            return A.HybridQNN(**kw)
        if name in ("dcqf", "dcqf_scrambled", "dcqf_product_control"):
            from qheart.models.heads import LogisticHead
            head_C = kw.pop("C", 1.0)
            tf = (ChainProductControl(**kw) if name == "dcqf_product_control"
                  else DCQFExtractor(scramble_couplings=(name == "dcqf_scrambled"), **kw))
            return _TransformThenHead(tf, LogisticHead(C=head_C))
        raise KeyError(name)
    return factory


class _TransformThenHead:
    """Fixed transform + shared head, standardising between them.

    Standardisation is not cosmetic here. DCQF outputs are bounded expectations in
    [-1, 1] while the chain-product control is unbounded before its tanh, and an L2
    penalty is scale-sensitive -- so without this the regulariser would penalise the
    two arms differently and the comparison would be measuring feature scale. The
    mean and sd are computed on the training rows only, inside ``fit``.
    """

    def __init__(self, transform, head):
        self.transform, self.head = transform, head
        self.mu_ = self.sd_ = None

    def fit(self, X, y):
        Z = self.transform.fit(X, y).transform(X)
        self.mu_ = Z.mean(axis=0)
        self.sd_ = np.where(Z.std(axis=0) < 1e-12, 1.0, Z.std(axis=0))
        self.head.fit((Z - self.mu_) / self.sd_, y)
        return self

    def predict_proba(self, X):
        if self.mu_ is None:
            raise RuntimeError("call fit before predicting")
        Z = self.transform.transform(X)
        return self.head.predict_proba((Z - self.mu_) / self.sd_)

    def n_quantum_params(self) -> int:
        return 0

    def n_trainable_params(self) -> int:
        return self.transform.n_trainable_params() + self.head.n_trainable_params()


REGISTRY: dict[str, ModelSpec] = {}


def _reg(s: ModelSpec) -> None:
    REGISTRY[s.name] = s


for _n, _f in __import__("qheart.models.classical", fromlist=["CLASSICAL"]).CLASSICAL.items():
    _reg(ModelSpec(_n, _f, note="classical baseline, preprocessing fitted inside the fold"))

_reg(ModelSpec("qkernel_zz", _qkernel, needs_quantum=True, input_kind="kernel_indices",
               note="ZZ fidelity kernel -> SVC; Gram matrix cached once, sliced per fold",
               defaults={"n_qubits": 4, "reps": 2, "entanglement": "linear"}))
_reg(ModelSpec("vqc_dense", _q("vqc"), needs_quantum=True, reports_params=True,
               note="variational classifier, dense angle encoding, 96 angles + 1 bias",
               defaults={"n_features": 8, "depth": 6, "epochs": 60}))
_reg(ModelSpec("hybrid_extractor", _q("hybrid_extractor"), needs_quantum=True,
               reports_params=True,
               note="quantum feature extractor + shared classical head (HQF-CC)",
               defaults={"n_features": 8, "depth": 6, "trainable": False}))
_reg(ModelSpec("control_c", _control, reports_params=True,
               note="PARAMETER-MATCHED CONTROL -- the comparison the papers omit; "
                    "same head as hybrid_extractor, classical reducer of equal size",
               defaults={"d_in": 8, "d_out": 4, "target_params": 96, "trainable": False}))

# --- NumPy-backend arms: the same three algorithms, executable without pennylane ---
_reg(ModelSpec("np_qsvc", _backend("np_qsvc"), reports_params=True,
               note="fidelity kernel + SVC (sklearn) or kernel logistic (fallback); "
                    "params grow as N+1, NOT the circuit angle count",
               defaults={"n_qubits": 4, "encoding": "dense_angle", "C": 1.0}))
_reg(ModelSpec("np_vqc", _backend("np_vqc"), reports_params=True,
               note="variational classifier trained by exact parameter-shift; the only "
                    "arm where the circuit angle count IS the model size",
               defaults={"n_qubits": 4, "depth": 6, "epochs": 30}))
_reg(ModelSpec("np_hybrid", _backend("np_hybrid"), reports_params=True,
               note="frozen quantum extractor + shared head (ADR-008): zero quantum "
                    "parameters, matched by control_c",
               defaults={"n_qubits": 4, "depth": 6, "trainable": False}))

# --- DCQF arm and its two mandatory controls. Never report the first without both. ---
_reg(ModelSpec("dcqf", _backend("dcqf"), reports_params=True,
               note="Kipu DCQF extractor + shared head; zero trainable quantum params",
               defaults={"orders_encoded": (2,), "orders_read": (1, 2, 3)}))
_reg(ModelSpec("dcqf_scrambled", _backend("dcqf_scrambled"), reports_params=True,
               note="NULL MODEL for dcqf: same circuit, couplings permuted off their "
                    "variables. Isolates whether the MI encoding (Eq. 2) does anything",
               defaults={"orders_encoded": (2,), "orders_read": (1, 2, 3)}))
_reg(ModelSpec("dcqf_product_control", _backend("dcqf_product_control"), reports_params=True,
               note="DIMENSION-MATCHED CONTROL for dcqf: classical products over the "
                    "same closed-chain subsets, same output width, no circuit",
               defaults={"orders": (1, 2, 3)}))


def available(include_quantum: bool = True) -> list[str]:
    return sorted(n for n, s in REGISTRY.items() if include_quantum or not s.needs_quantum)


def quantum_models() -> list[str]:
    return sorted(n for n, s in REGISTRY.items() if s.needs_quantum)


def classical_models() -> list[str]:
    return sorted(n for n, s in REGISTRY.items() if not s.needs_quantum)


def spec(name: str) -> ModelSpec:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}. Available: {available()}")
    return REGISTRY[name]


def build(name: str, **overrides):
    """Return a *factory* -- ``build(name)()`` makes a fresh instance.

    A factory, not an instance, because the harness needs a new model per fold. A
    single shared instance would carry state across folds and quietly turn 25
    independent fits into one long one.
    """
    s = spec(name)
    kw = {**s.defaults, **overrides}
    return lambda: s.factory(**kw)


def describe() -> str:
    w = max(len(n) for n in REGISTRY)
    lines = [f"{'model'.ljust(w)}  {'quantum':8s} {'input':16s} note", "-" * 108]
    for n in sorted(REGISTRY, key=lambda k: (REGISTRY[k].needs_quantum, k)):
        s = REGISTRY[n]
        lines.append(f"{n.ljust(w)}  {'yes' if s.needs_quantum else 'no':8s} "
                     f"{s.input_kind:16s} {s.note}")
    return "\n".join(lines)
