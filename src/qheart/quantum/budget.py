"""The O(N^2) cost guard. Pure Python -- no quantum dependency, so it can be
consulted before committing to a run.

A quantum kernel needs one circuit evaluation per *pair* of samples. At 918 rows a
full Gram matrix is ~421k evaluations; the same code pointed at a 20,000-row dataset
is ~2x10^8 and will not finish before the deadline. This module exists so that fact
is discovered by reading a printed estimate, not by watching a progress bar for two
days.

``guard_kernel_size`` raises by default. A hard failure at minute zero costs nothing;
a soft warning gets scrolled past.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KernelBudget", "estimate_kernel_cost", "guard_kernel_size", "fmt_seconds"]


@dataclass(frozen=True)
class KernelBudget:
    n_train: int
    n_test: int
    train_evals: int
    test_evals: int
    total_evals: int
    seconds_per_eval: float
    est_seconds: float
    folds: int

    def report(self) -> str:
        return (
            f"Quantum kernel cost estimate\n"
            f"  train Gram   {self.n_train} x {self.n_train} -> {self.train_evals:,} evaluations"
            f" (upper triangle only)\n"
            f"  test block   {self.n_test} x {self.n_train} -> {self.test_evals:,} evaluations\n"
            f"  per fold     {self.total_evals:,} evaluations\n"
            f"  x {self.folds} folds  -> {self.total_evals * self.folds:,} evaluations\n"
            f"  at {self.seconds_per_eval * 1000:.2f} ms each -> "
            f"{fmt_seconds(self.est_seconds * self.folds)}\n"
            f"  NOTE the full-dataset Gram matrix can be computed ONCE and sliced per\n"
            f"  fold, which removes the x{self.folds}. Do that; see kernel.py cache."
        )


def fmt_seconds(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    if s < 172800:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


def estimate_kernel_cost(n_train: int, n_test: int = 0, seconds_per_eval: float = 0.004,
                         folds: int = 1) -> KernelBudget:
    """Circuit evaluations for a kernel run.

    The training Gram matrix is symmetric with a known unit diagonal, so only the
    strict upper triangle needs evaluating: ``n(n-1)/2``, not ``n^2``. That halving
    is the single cheapest optimisation available here and is why the default
    ``seconds_per_eval`` of 4 ms (a statevector simulator, 8 qubits, reps=2, no
    shots) is worth measuring rather than guessing.
    """
    if n_train < 0 or n_test < 0:
        raise ValueError("sample counts must be non-negative")
    tr = n_train * (n_train - 1) // 2
    te = n_test * n_train
    total = tr + te
    return KernelBudget(n_train=n_train, n_test=n_test, train_evals=tr, test_evals=te,
                        total_evals=total, seconds_per_eval=seconds_per_eval,
                        est_seconds=total * seconds_per_eval, folds=folds)


def guard_kernel_size(n_samples: int, max_samples: int = 2000, hard: bool = True) -> None:
    """Refuse a kernel run that cannot finish. Raises unless ``hard=False``.

    2000 is not a law of nature -- it is roughly where a simulated Gram matrix stops
    fitting in an evening on a laptop. Raise the ceiling deliberately, in the config,
    with the estimate in front of you.
    """
    if n_samples <= max_samples:
        return
    est = estimate_kernel_cost(n_samples)
    msg = (f"quantum kernel on {n_samples} samples needs {est.total_evals:,} circuit "
           f"evaluations (~{fmt_seconds(est.est_seconds)} simulated). Ceiling is "
           f"{max_samples}. Either subsample, or raise max_samples in the config and "
           f"accept the runtime.")
    if hard:
        raise ValueError(msg)
    print("WARNING: " + msg)
