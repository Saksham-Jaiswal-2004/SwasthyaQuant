"""Quantum layer. PennyLane for models, Qiskit for hardware execution and noise.

Every import of pennylane/qiskit in this subpackage is *inside a function*. That is
not stylistic: it means the classical half of the project keeps running when the
quantum install breaks, which it will, at least once, probably the week before
submission. A broken `pip install qiskit-aer` should never stop someone from
producing the baseline table.
"""

from qheart.quantum.budget import KernelBudget, estimate_kernel_cost, guard_kernel_size

__all__ = ["KernelBudget", "estimate_kernel_cost", "guard_kernel_size"]
