"""Dimensionality reduction to the qubit budget, fitted inside the fold.

Retained variance is logged next to every result. A quantum model fed 8 components
holding 61% of the variance and a classical model fed all 11 features are not
solving the same problem, and reporting them side by side without the variance
figure is the kind of comparison that falls apart under one question.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_reducer", "retained_variance"]


def build_reducer(kind: str = "pca", n_components: int = 8, seed: int = 20260830):
    """An *unfitted* reducer. sklearn imported lazily, as everywhere else here."""
    if kind == "none":
        return None
    if kind == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=seed)
    if kind == "kernel_pca":
        from sklearn.decomposition import KernelPCA
        return KernelPCA(n_components=n_components, kernel="rbf", random_state=seed)
    raise ValueError(f"unknown reducer {kind!r}; use 'pca', 'kernel_pca' or 'none'")


def retained_variance(reducer) -> float | None:
    """Fraction of variance retained, or None when the reducer cannot report it.

    Returns None rather than 0.0 for KernelPCA. A zero would be logged as a real
    measurement and read as catastrophic; None reads as "not available", which is
    the truth.
    """
    r = getattr(reducer, "explained_variance_ratio_", None)
    return None if r is None else float(np.sum(r))
