"""Parameter accounting. The centre of the whole project.

The claim we intend to defend is not "quantum is more accurate" -- it probably is
not. It is "a circuit with a few dozen trainable angles matches a classical model
with far more parameters". That claim is only worth anything if the parameter
counts are computed honestly and the comparison is matched, so counting lives in
its own module with its own tests instead of being inlined somewhere.

Two rules enforced here:

1. **A gate with no angle is not a parameter.** Reference paper P1 counts its CNOTs
   as trainable parameters, which inflates 16 angles per layer to 19 and turns 96
   into 114. CNOT has no continuous parameter to learn. ``count_circuit_params``
   counts rotation angles only.

2. **Control-C is sized FROM the circuit at runtime**, never hard-coded. If someone
   changes the depth from 6 to 8, the control resizes itself on the next run. A
   hard-coded control silently stops being matched the first time the circuit
   changes, and nobody notices because the number still looks plausible.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CircuitParamCount", "ControlSizing", "count_circuit_params",
           "size_control_reducer", "count_torch_params", "bytes_at", "assert_matched"]


@dataclass(frozen=True)
class CircuitParamCount:
    n_qubits: int
    depth: int
    rotations_per_qubit_per_layer: int
    trainable: int
    entanglers: int
    note: str = ""

    def bytes_fp32(self) -> int:
        return self.trainable * 4

    def __str__(self) -> str:
        return (f"{self.n_qubits} qubits x depth {self.depth} -> {self.trainable} "
                f"trainable angles ({self.bytes_fp32()} bytes at fp32); "
                f"{self.entanglers} entangling gates, which are NOT parameters")


def count_circuit_params(n_qubits: int, depth: int, rotations_per_qubit_per_layer: int = 4,
                         entanglers_per_layer: int | None = None) -> CircuitParamCount:
    """Trainable angles in a layered variational ansatz.

    Defaults reproduce the P1 geometry: 4 qubits, 4 rotations per qubit per layer
    (RY, RX, RZ, RY), so 16 angles per layer -- 64 at depth 4 and 96 at depth 6.
    Entanglers are counted and reported separately, deliberately, so nobody folds
    them back into the parameter total.
    """
    if n_qubits < 1 or depth < 1:
        raise ValueError("n_qubits and depth must be >= 1")
    per_layer = n_qubits * rotations_per_qubit_per_layer
    ent = (entanglers_per_layer if entanglers_per_layer is not None else n_qubits) * depth
    return CircuitParamCount(
        n_qubits=n_qubits, depth=depth,
        rotations_per_qubit_per_layer=rotations_per_qubit_per_layer,
        trainable=per_layer * depth, entanglers=ent,
        note="entangling gates carry no trainable angle and are excluded",
    )


@dataclass(frozen=True)
class ControlSizing:
    target: int
    d_in: int
    d_out: int
    hidden_below: int
    params_below: int
    hidden_above: int
    params_above: int

    @property
    def exact(self) -> bool:
        return self.params_below == self.target

    def summary(self) -> str:
        if self.exact:
            return (f"Control-C matches exactly: hidden={self.hidden_below}, "
                    f"{self.params_below} params vs {self.target} quantum angles")
        return (f"Control-C brackets the circuit: hidden={self.hidden_below} gives "
                f"{self.params_below} params (below {self.target}) and "
                f"hidden={self.hidden_above} gives {self.params_above} (above). "
                f"Run both -- then the quantum model cannot be accused of winning "
                f"on capacity, whichever way the result falls.")


def size_control_reducer(target: int, d_in: int, d_out: int) -> ControlSizing:
    """Size a classical ``d_in -> h -> d_out`` MLP to the circuit's parameter budget.

    Parameter count is ``h*(d_in + 1 + d_out) + d_out`` (two weight matrices, two
    bias vectors). Integer ``h`` rarely hits the target exactly, so this returns the
    nearest width below and the nearest above. Run both controls: if the quantum
    model beats the smaller control but loses to the larger one, the honest
    conclusion is that the comparison is a wash, and reporting that is what makes
    the rest of the work credible.
    """
    if target < 1 or d_in < 1 or d_out < 1:
        raise ValueError("target, d_in and d_out must all be >= 1")
    per_hidden = d_in + 1 + d_out

    def n_params(h: int) -> int:
        return h * per_hidden + d_out

    h = max(1, (target - d_out) // per_hidden)
    while n_params(h + 1) <= target:
        h += 1
    while h > 1 and n_params(h) > target:
        h -= 1
    return ControlSizing(target=target, d_in=d_in, d_out=d_out,
                         hidden_below=h, params_below=n_params(h),
                         hidden_above=h + 1, params_above=n_params(h + 1))


def count_torch_params(module) -> int:
    """Trainable parameters in a torch module. Frozen tensors are excluded, which is
    the point when a pretrained backbone is frozen and only the head is learned."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def bytes_at(n_params: int, dtype: str = "fp32") -> int:
    width = {"fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8}
    if dtype not in width:
        raise ValueError(f"unknown dtype {dtype!r}")
    return n_params * width[dtype]


def assert_matched(n_quantum: int, n_control: int, tolerance: float = 0.05) -> None:
    """Fail loudly if a 'parameter-matched' control is not actually matched.

    Call this in the run path, not just in tests. The failure mode this prevents is
    a control that drifts after an unrelated config change and quietly becomes a
    different experiment while still being labelled the control.
    """
    if n_quantum <= 0:
        raise ValueError("quantum parameter count must be positive")
    rel = abs(n_control - n_quantum) / n_quantum
    if rel > tolerance:
        raise AssertionError(
            f"control has {n_control} trainable parameters vs {n_quantum} in the "
            f"circuit ({rel:.1%} off, tolerance {tolerance:.0%}). Either resize the "
            f"control with size_control_reducer() or stop calling it matched."
        )
