"""Parameter-shift saliency: the only explanation that is actually about the circuit.

For a circuit output f(x), the derivative with respect to input angle x_i is obtained
exactly -- not by finite differences -- as

    df/dx_i = [ f(x + (pi/2) e_i) - f(x - (pi/2) e_i) ] / 2

which is the parameter-shift rule applied to an *input* angle rather than a weight.
Two circuit evaluations per feature per sample, so cost is 2*d*n. At d=8 and n=200
that is 3,200 evaluations: seconds on a simulator, and the reason this is only ever
run on a subsample when shots are finite.

Caveat to state in the caption: this is a local gradient. It says how the output
responds to a small change around x, not how important the feature is globally, and
where the circuit output saturates the gradient goes to zero even though the feature
matters. Average magnitudes over a sample and label them as such.
"""

from __future__ import annotations

import numpy as np

__all__ = ["circuit_saliency", "saliency_note", "aggregate_saliency"]

SHIFT = np.pi / 2


def saliency_note() -> str:
    return ("Parameter-shift saliency: exact local derivative of the circuit output "
            "with respect to each input angle, averaged in absolute value over the "
            "sample. This explains the CIRCUIT. It is a local gradient, so a "
            "saturated response reads as zero importance -- read it alongside the SHAP "
            "figure for the head, and do not merge the two.")


def circuit_saliency(circuit, x: np.ndarray, weights, encoding_dim: int | None = None
                     ) -> np.ndarray:
    """Exact per-feature derivative at a single point ``x``."""
    x = np.asarray(x, dtype=float)
    d = encoding_dim or x.size
    g = np.zeros(d)
    for i in range(d):
        xp, xm = x.copy(), x.copy()
        xp[i] += SHIFT
        xm[i] -= SHIFT
        g[i] = (float(np.ravel(circuit(xp, weights))[0])
                - float(np.ravel(circuit(xm, weights))[0])) / 2.0
    return g


def aggregate_saliency(circuit, X: np.ndarray, weights, feature_names: list[str] | None = None,
                       max_samples: int = 200, seed: int = 20260830):
    """Mean |gradient| per feature over a subsample, plus the evaluation count.

    The evaluation count is returned, not just printed, because "how expensive is your
    explanation" is a fair question and an answer with a number in it lands better
    than one without.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    n = min(max_samples, X.shape[0])
    idx = rng.choice(X.shape[0], size=n, replace=False)
    G = np.array([circuit_saliency(circuit, X[i], weights) for i in idx])
    names = feature_names or [f"f{i}" for i in range(X.shape[1])]
    order = np.argsort(np.abs(G).mean(axis=0))[::-1]
    return ([(names[i], float(np.abs(G).mean(axis=0)[i])) for i in order],
            {"circuit_evaluations": int(2 * X.shape[1] * n), "n_samples": n})
