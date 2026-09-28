"""Noise and hardware. The section that answers "would this survive a real device?"

The deliverable is a **degradation curve**: metric on the vertical axis, shot count
and two-qubit error rate on the horizontal. One plot that shows the model degrading
gracefully is worth more than a single hardware run that happened to succeed, and it
cannot be dismissed as a lucky queue slot.

Order of work, and it matters: simulator -> noisy simulator -> one hardware job, in
that order, with the hardware job never on the critical path. Free-tier IBM queues
are measured in hours and the deadline is not negotiable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NoiseSpec", "build_noise_model", "shot_grid", "degradation_curve",
           "runtime_sampler", "SHOT_GRID"]

# Log-spaced, because the interesting behaviour is the 1/sqrt(shots) regime at the
# low end. Ten evenly spaced values from 1024 to 8192 would show almost nothing.
SHOT_GRID = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)


@dataclass(frozen=True)
class NoiseSpec:
    """Depolarising + readout error, roughly calibrated to a small IBM device.

    Defaults are deliberately pessimistic relative to the best available hardware.
    Being pleasantly surprised by a real device is a good outcome; claiming
    robustness from an optimistic noise model and then failing on hardware is not.
    """
    p_1q: float = 0.001
    p_2q: float = 0.01
    p_readout: float = 0.02
    label: str = "small-device-pessimistic"

    def describe(self) -> str:
        return (f"{self.label}: 1-qubit depolarising {self.p_1q:.4f}, 2-qubit "
                f"{self.p_2q:.4f}, readout flip {self.p_readout:.3f}")


def build_noise_model(spec: NoiseSpec | None = None):
    """A qiskit-aer NoiseModel. Imported lazily -- aer is the most fragile dependency
    in the stack and must not be able to break the classical path."""
    from qiskit_aer.noise import (NoiseModel, ReadoutError, depolarizing_error)

    spec = spec or NoiseSpec()
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(spec.p_1q, 1),
                                   ["u1", "u2", "u3", "rz", "sx", "x", "ry", "rx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(spec.p_2q, 2), ["cx", "cz"])
    r = spec.p_readout
    nm.add_all_qubit_readout_error(ReadoutError([[1 - r, r], [r, 1 - r]]))
    return nm


def shot_grid(grid=SHOT_GRID) -> list[int]:
    return list(grid)


def degradation_curve(score_fn, shots_grid=SHOT_GRID, repeats: int = 3,
                      seed: int = 20260830) -> "list[dict]":
    """Evaluate ``score_fn(shots, seed)`` across the shot grid.

    ``repeats`` matters: a single measurement at 32 shots is itself a noisy estimate,
    so a curve built from one run per point will look jagged and invite the reading
    that the model is unstable, when the *estimator* is what is unstable. Report
    mean and spread at each point.
    """
    from qheart.seeds import derive

    rows = []
    for shots in shots_grid:
        vals = [float(score_fn(shots, derive(seed, "noise", str(shots), str(r))))
                for r in range(repeats)]
        rows.append({"shots": int(shots), "mean": float(np.mean(vals)),
                     "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                     "min": float(np.min(vals)), "max": float(np.max(vals)),
                     "repeats": repeats})
    return rows


def runtime_sampler(backend_name: str | None = None, shots: int = 1024,
                    simulator_fallback: bool = True):
    """An IBM Runtime Sampler, falling back to the local simulator by default.

    The fallback is not laziness. It means a queue outage on submission weekend
    degrades one supporting figure instead of breaking the pipeline. Whatever runs,
    the caller must record which -- a plot labelled "hardware" that was produced by a
    simulator is the one mistake here that would be genuinely dishonest.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        service = QiskitRuntimeService()
        backend = (service.backend(backend_name) if backend_name
                   else service.least_busy(operational=True, simulator=False))
        print(f"IBM Runtime: using {backend.name} "
              f"(queue length {getattr(backend.status(), 'pending_jobs', '?')})")
        return SamplerV2(mode=backend), backend.name
    except Exception as exc:                                  # noqa: BLE001
        if not simulator_fallback:
            raise
        print(f"IBM Runtime unavailable ({type(exc).__name__}: {exc}). "
              f"Falling back to the local simulator. Label the figure 'simulator'.")
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        return AerSampler(default_shots=shots), "aer_simulator"
