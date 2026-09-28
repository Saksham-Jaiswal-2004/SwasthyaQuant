"""Variational ansatz. Where the trainable parameters live, and where the barren
plateaus live too.

Shape convention is ``(depth, n_qubits, rotations_per_qubit)``, so the parameter
count is the product and ``qheart.params.count_circuit_params`` can be checked
against ``weights.size`` at runtime -- which is exactly what
``assert_param_count`` does below. The two must never be allowed to drift apart,
because the parameter count is the headline claim.
"""

from __future__ import annotations

import numpy as np

from qheart.params import count_circuit_params

__all__ = ["weight_shape", "init_weights", "layered_ansatz", "assert_param_count",
           "ENTANGLERS"]

ENTANGLERS = ("ring", "linear", "all_to_all", "none")
ROTATIONS_PER_QUBIT = 4          # RY, RX, RZ, RY -- the P1 geometry


def weight_shape(n_qubits: int, depth: int,
                 rotations_per_qubit: int = ROTATIONS_PER_QUBIT) -> tuple[int, int, int]:
    return (depth, n_qubits, rotations_per_qubit)


def init_weights(n_qubits: int, depth: int, seed: int = 20260830,
                 rotations_per_qubit: int = ROTATIONS_PER_QUBIT,
                 scale: float = 0.1) -> np.ndarray:
    """Small random initial angles.

    ``scale=0.1`` rather than uniform over [0, 2*pi] on purpose. Wide random
    initialisation in a deep circuit lands in a barren plateau where gradients are
    exponentially small in the qubit count and training stalls at chance with a
    perfectly flat, perfectly healthy-looking loss curve. Starting near the identity
    keeps early gradients measurable. If the loss is flat from step 1, lower this
    before touching anything else.
    """
    from qheart.seeds import spawn
    rng = spawn(seed, "ansatz_init")
    return rng.normal(0.0, scale, size=weight_shape(n_qubits, depth, rotations_per_qubit))


def assert_param_count(weights, n_qubits: int, depth: int,
                       rotations_per_qubit: int = ROTATIONS_PER_QUBIT) -> int:
    """Cross-check the tensor against the analytic count. Returns the count.

    Called on every fit. The failure this catches is a reshape bug that leaves the
    model training happily with 64 parameters while the report claims 96 -- the model
    still works, so nothing else would ever notice.
    """
    expected = count_circuit_params(n_qubits, depth, rotations_per_qubit).trainable
    actual = int(np.asarray(weights).size)
    if actual != expected:
        raise AssertionError(
            f"ansatz holds {actual} angles but the analytic count says {expected}. "
            f"One of the two is wrong, and the parameter count is the claim this "
            f"project is built on -- fix it before reporting anything."
        )
    return expected


def _entangle(kind: str, wires) -> None:
    import pennylane as qml
    n = len(wires)
    if kind == "none" or n < 2:
        return
    if kind == "linear":
        for i in range(n - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
    elif kind == "ring":
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])
    elif kind == "all_to_all":
        # O(n^2) two-qubit gates. On real hardware every non-adjacent CNOT is
        # decomposed into a chain of SWAPs, so this is far more expensive than it
        # looks in the diagram -- expect the noise curve to collapse.
        for i in range(n):
            for j in range(i + 1, n):
                qml.CNOT(wires=[wires[i], wires[j]])
    else:
        raise ValueError(f"unknown entangler {kind!r}; choose from {ENTANGLERS}")


def layered_ansatz(weights, wires, entangler: str = "ring") -> None:
    """``depth`` blocks of single-qubit rotations followed by entanglement.

    Rotations then entanglers, in that order, every layer. Entangling first on layer
    0 does nothing useful: the state is still a product state at that point, so the
    CNOTs act on a basis state and generate no correlation to exploit.
    """
    import pennylane as qml
    w = np.asarray(weights) if not hasattr(weights, "shape") else weights
    depth = w.shape[0]
    gates = (qml.RY, qml.RX, qml.RZ, qml.RY)
    for d in range(depth):
        for q, wire in enumerate(wires):
            for r in range(w.shape[2]):
                gates[r % len(gates)](w[d, q, r], wires=wire)
        _entangle(entangler, wires)
