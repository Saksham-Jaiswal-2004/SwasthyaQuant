"""Encodings and ansatz circuits on the NumPy statevector backend.

``qheart.quantum.encodings`` and ``qheart.quantum.ansatz`` describe the same circuits
for PennyLane, by queuing gates against a global tape. That is the right thing when
PennyLane is installed and the target is hardware. This module is the executable
counterpart: the identical circuit geometry expressed against
``qheart.quantum.statevector.State``, so every model in this repo can be fitted,
tested and demoed with numpy alone.

**The two must not drift apart.** The angle conventions here (RY for angle encoding,
the RY-RX-RZ-RY rotation block, the ring entangler order) mirror
``ROTATIONS_PER_QUBIT`` and ``_entangle`` exactly, and ``tests/`` compares parameter
counts against ``ansatz.weight_shape`` so a change to one side fails loudly rather
than quietly producing two different "4-qubit, depth-6" models.

On the fidelity kernel: on hardware you get |<phi(a)|phi(b)>|^2 from a
compute-uncompute circuit, preparing phi(a), applying the inverse of phi(b), and
measuring the probability of the all-zeros string. In exact simulation the direct
inner product of the two state vectors is the same number, computed in one dot product
instead of a circuit twice the depth. ``fidelity`` therefore takes the shortcut and
says so; ``fidelity_compute_uncompute`` runs the hardware version and exists so a test
can prove the two agree.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from qheart.quantum.ansatz import ROTATIONS_PER_QUBIT, weight_shape
from qheart.quantum.encodings import ENCODINGS, check_range, n_qubits_for
from qheart.quantum.statevector import State, guard_qubits

__all__ = ["encode", "apply_ansatz", "prepare", "fidelity", "gram_matrix",
           "fidelity_compute_uncompute", "n_weights"]


# ------------------------------------------------------------------ encodings
def encode(st: State, x: np.ndarray, encoding: str = "dense_angle") -> State:
    """Apply a feature-encoding block in place.

    ``angle``       : one feature per qubit, RY(x_i).
    ``dense_angle`` : two features per qubit, RY(x_2i) then RZ(x_2i+1) -- halves the
                      qubit count, which is what makes 8 clinical features fit on 4
                      qubits. Note the second feature enters as a *phase*, so it is
                      invisible to a Z-basis measurement unless something later rotates
                      it into the computational basis. The ansatz does; a bare
                      dense-angle circuit read out in Z would silently ignore half the
                      data, and this is the single most common way to get a
                      "quantum model ignores my features" bug.
    ``amplitude``   : the normalised feature vector as amplitudes. Encodes 2^n features
                      on n qubits, but the state-preparation depth is the honest
                      objection and normalisation destroys overall magnitude.

    Inputs must already sit in [0, pi]; ``check_range`` raises rather than clipping,
    because silently clipping a mis-scaled column turns a preprocessing bug into a
    mediocre result instead of an error.
    """
    x = np.asarray(x, dtype=float).ravel()
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; choose from {ENCODINGS}")

    if encoding == "amplitude":
        need = 2 ** st.n
        if x.size > need:
            raise ValueError(f"amplitude encoding on {st.n} qubits holds {need} values, got {x.size}")
        v = np.zeros(need, dtype=np.complex128)
        v[:x.size] = x
        nrm = np.linalg.norm(v)
        if nrm <= 0:
            raise ValueError("cannot amplitude-encode an all-zero feature vector")
        st.vec = v / nrm
        return st

    check_range(x)
    if encoding == "angle":
        if x.size != st.n:
            raise ValueError(f"angle encoding needs one feature per qubit: {x.size} vs {st.n}")
        for i, xi in enumerate(x):
            st.ry(i, float(xi))
        return st

    # dense_angle
    if x.size > 2 * st.n:
        raise ValueError(f"dense-angle on {st.n} qubits holds {2*st.n} features, got {x.size}")
    st.h_all()                       # start from |+>^n so RZ phases are observable
    for i in range(st.n):
        j = 2 * i
        if j < x.size:
            st.ry(i, float(x[j]))
        if j + 1 < x.size:
            st.rz(i, float(x[j + 1]))
    return st


# --------------------------------------------------------------------- ansatz
def n_weights(n_qubits: int, depth: int) -> int:
    """Trainable angle count. Delegates to ``ansatz.weight_shape`` so there is exactly
    one definition of this number in the codebase -- the figure that ends up on a slide
    must not be computed in two places."""
    return int(np.prod(weight_shape(n_qubits, depth)))


def _entangle_pairs(kind: str, n: int) -> list[tuple[int, int]]:
    """CNOT pairs for one layer. Entanglers carry NO trainable parameters.

    Counting them as parameters is the specific arithmetic error this project
    documented in Paper 1 (19 per layer where 16 is correct). They are reported
    separately by ``circuit_cost`` and never summed into a parameter total.
    """
    if kind == "none" or n < 2:
        return []
    if kind == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if kind == "ring":
        return [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]
    if kind == "all_to_all":
        return [(a, b) for a in range(n) for b in range(a + 1, n)]
    raise ValueError(f"unknown entangler {kind!r}")


def apply_ansatz(st: State, weights: np.ndarray, entangler: str = "ring") -> State:
    """Layered RY-RX-RZ-RY rotations with an entangling block per layer.

    ``weights`` has shape (depth, n_qubits, 4), matching ``ansatz.weight_shape``.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 3 or w.shape[2] != ROTATIONS_PER_QUBIT:
        raise ValueError(
            f"weights must be (depth, n_qubits, {ROTATIONS_PER_QUBIT}), got {w.shape}"
        )
    if w.shape[1] != st.n:
        raise ValueError(f"weights are for {w.shape[1]} qubits, state has {st.n}")
    for layer in w:
        for q in range(st.n):
            st.ry(q, float(layer[q, 0]))
            st.rx(q, float(layer[q, 1]))
            st.rz(q, float(layer[q, 2]))
            st.ry(q, float(layer[q, 3]))
        for a, b in _entangle_pairs(entangler, st.n):
            st.cnot(a, b)
    return st


def prepare(x: np.ndarray, n_qubits: int, encoding: str = "dense_angle",
            weights: np.ndarray | None = None, entangler: str = "ring") -> State:
    """Encoding, then optionally the ansatz. The one place the circuit is assembled."""
    guard_qubits(n_qubits)
    st = State(n_qubits)
    encode(st, x, encoding)
    if weights is not None:
        apply_ansatz(st, weights, entangler)
    return st


# --------------------------------------------------------------------- kernel
def fidelity(a: State, b: State) -> float:
    """|<a|b>|^2, the exact-simulation shortcut for the compute-uncompute kernel."""
    ov = complex(np.vdot(a.vec, b.vec))
    return float(np.abs(ov) ** 2)


def fidelity_compute_uncompute(x: np.ndarray, z: np.ndarray, n_qubits: int,
                               encoding: str = "dense_angle") -> float:
    """The hardware construction: prepare phi(x), undo phi(z), read P(|0...0>).

    Kept so a test can show it equals ``fidelity``. Do not use it in a fitting loop --
    it is twice the depth for an identical number.
    """
    st = prepare(x, n_qubits, encoding)
    inv = prepare(z, n_qubits, encoding)
    # <0|U(z)^dag U(x)|0> = <U(z)0 | U(x)0>; probability of all-zeros after inversion.
    return float(np.abs(complex(np.vdot(inv.vec, st.vec))) ** 2)


def gram_matrix(A: np.ndarray, B: np.ndarray | None = None, n_qubits: int = 4,
                encoding: str = "dense_angle", symmetric: bool | None = None
                ) -> np.ndarray:
    """Fidelity Gram matrix. States are prepared ONCE per row, not once per pair.

    This is the difference between O(N) and O(N^2) state preparations. The kernel is
    still O(N^2) *entries*, which is the real scaling problem: at 70,000 patients that
    is 2.45e9 entries and roughly 113 days at this repo's measured rate, which is why
    kernel arms run on a pre-registered stratified subsample and say so.
    """
    A = np.asarray(A, dtype=float)
    sa = [prepare(r, n_qubits, encoding) for r in A]
    if B is None:
        sb, symmetric = sa, True if symmetric is None else symmetric
    else:
        B = np.asarray(B, dtype=float)
        sb = [prepare(r, n_qubits, encoding) for r in B]
        symmetric = False if symmetric is None else symmetric

    Va = np.stack([s.vec for s in sa])
    Vb = Va if B is None else np.stack([s.vec for s in sb])
    K = np.abs(Va.conj() @ Vb.T) ** 2
    if symmetric:
        K = 0.5 * (K + K.T)          # kill float asymmetry at 1e-16, not real skew
        np.fill_diagonal(K, 1.0)     # <phi|phi> = 1 exactly, by normalisation
    return np.asarray(K, dtype=float)
