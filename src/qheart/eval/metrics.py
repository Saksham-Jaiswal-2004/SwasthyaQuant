"""Metrics, in pure numpy.

No sklearn dependency on purpose. These are the numbers the whole project will be
judged on, and they should not change because someone upgraded a library. The
implementations are checked against hand-computed values in tests/test_metrics.py.

Ordering matters here. ``HEADLINE`` is sensitivity and specificity, and the report
builder refuses to emit a table that omits them. That is a deliberate constraint:
accuracy on a 55/45 split is nearly uninformative -- a model that always predicts
"disease" already scores about 0.55 -- while a model with 0.96 accuracy and 0.70
sensitivity misses three in ten sick patients. If only one number can be quoted,
it must not be accuracy.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "HEADLINE", "SECONDARY", "ALL_METRICS",
    "confusion", "sensitivity", "specificity", "ppv", "npv",
    "accuracy", "balanced_accuracy", "f1", "mcc", "cohen_kappa", "youden_j",
    "roc_auc", "average_precision", "brier", "expected_calibration_error",
    "summary", "aggregate", "best_threshold_youden", "threshold_at_sensitivity",
    "bootstrap_ci",
]

# Quoted first, always. See module docstring.
HEADLINE = ["sensitivity", "specificity"]
SECONDARY = ["ppv", "npv", "pr_auc", "roc_auc", "balanced_accuracy",
             "f1", "mcc", "kappa", "brier", "ece", "accuracy"]
ALL_METRICS = HEADLINE + SECONDARY

_EPS = 1e-12


def _as_binary(y) -> np.ndarray:
    y = np.asarray(y).ravel()
    u = np.unique(y)
    if not np.all(np.isin(u, [0, 1])):
        raise ValueError(f"labels must be 0/1, got {u.tolist()}")
    return y.astype(int)


def confusion(y_true, y_pred) -> tuple[int, int, int, int]:
    """Return ``(tn, fp, fn, tp)``. Note the order -- it matches sklearn's ravel()."""
    y = _as_binary(y_true)
    p = _as_binary(y_pred)
    if y.size != p.size:
        raise ValueError(f"length mismatch: {y.size} vs {p.size}")
    tp = int(np.sum((y == 1) & (p == 1)))
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    return tn, fp, fn, tp


def sensitivity(y_true, y_pred) -> float:
    """TP / (TP + FN). Recall. The fraction of sick patients we catch."""
    _, _, fn, tp = confusion(y_true, y_pred)
    return float(tp / (tp + fn)) if (tp + fn) else float("nan")


def specificity(y_true, y_pred) -> float:
    """TN / (TN + FP). The fraction of healthy patients we do not alarm."""
    tn, fp, _, _ = confusion(y_true, y_pred)
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def ppv(y_true, y_pred) -> float:
    """Precision. Depends on prevalence -- do not compare across datasets."""
    _, fp, _, tp = confusion(y_true, y_pred)
    return float(tp / (tp + fp)) if (tp + fp) else float("nan")


def npv(y_true, y_pred) -> float:
    tn, _, fn, _ = confusion(y_true, y_pred)
    return float(tn / (tn + fn)) if (tn + fn) else float("nan")


def accuracy(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion(y_true, y_pred)
    n = tn + fp + fn + tp
    return float((tp + tn) / n) if n else float("nan")


def balanced_accuracy(y_true, y_pred) -> float:
    return float(0.5 * (sensitivity(y_true, y_pred) + specificity(y_true, y_pred)))


def f1(y_true, y_pred) -> float:
    p, r = ppv(y_true, y_pred), sensitivity(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) > _EPS else 0.0


def mcc(y_true, y_pred) -> float:
    """Matthews correlation. The most honest single number under imbalance."""
    tn, fp, fn, tp = confusion(y_true, y_pred)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > _EPS else 0.0


def cohen_kappa(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion(y_true, y_pred)
    n = tn + fp + fn + tp
    if not n:
        return float("nan")
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n)
    return float((po - pe) / (1 - pe)) if abs(1 - pe) > _EPS else 0.0


def youden_j(y_true, y_pred) -> float:
    return float(sensitivity(y_true, y_pred) + specificity(y_true, y_pred) - 1.0)


def _rank_average(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties averaged. Equivalent to scipy.stats.rankdata."""
    x = np.asarray(x, dtype=float)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc(y_true, y_score) -> float:
    """ROC-AUC via the Mann-Whitney U statistic, with correct tie handling.

    Under class imbalance prefer ``average_precision``: ROC-AUC is computed against
    the negative class, so it stays flattering even when precision is poor.
    """
    y = _as_binary(y_true)
    s = np.asarray(y_score, dtype=float).ravel()
    if y.size != s.size:
        raise ValueError("length mismatch")
    n_pos = int(y.sum()); n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rank_average(s)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true, y_score) -> float:
    """Area under the precision-recall curve, step interpolation.

    Matches sklearn's ``average_precision_score``: sum over thresholds of
    (R_n - R_{n-1}) * P_n. Prefer this to ROC-AUC as the single ranking metric,
    and note its baseline is the prevalence, not 0.5.
    """
    y = _as_binary(y_true)
    s = np.asarray(y_score, dtype=float).ravel()
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    # Only evaluate at the last index of each run of equal scores, otherwise ties
    # are split arbitrarily and the number depends on row order.
    last = np.r_[np.flatnonzero(np.diff(ss)), ys.size - 1]
    tp = np.cumsum(ys)[last]
    fp = (last + 1) - tp
    precision = tp / np.maximum(tp + fp, _EPS)
    recall = tp / n_pos
    recall = np.r_[0.0, recall]
    return float(np.sum(np.diff(recall) * precision))


def brier(y_true, y_prob) -> float:
    """Mean squared error of the probabilities. Lower is better; 0.25 = uninformative.

    Include it because a model can rank well (good AUC) and still be badly
    calibrated, which matters the moment a clinician reads the number as a risk.
    """
    y = _as_binary(y_true)
    p = np.asarray(y_prob, dtype=float).ravel()
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Equal-width binned |accuracy - confidence|, weighted by bin size."""
    y = _as_binary(y_true)
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / y.size) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def best_threshold_youden(y_true, y_score) -> float:
    """Threshold maximising sensitivity + specificity - 1.

    Choose it on training folds only. Picking a threshold on the test set is a
    quiet, very common form of overfitting.
    """
    y = _as_binary(y_true)
    s = np.asarray(y_score, dtype=float).ravel()
    cand = np.unique(s)
    best, best_j = 0.5, -np.inf
    for t in cand:
        j = youden_j(y, (s >= t).astype(int))
        if j > best_j:
            best_j, best = j, float(t)
    return best


def threshold_at_sensitivity(y_true, y_score, target: float = 0.90) -> float:
    """Lowest threshold whose sensitivity is at least ``target``.

    Screening framing: fix the miss rate you can clinically tolerate, then report
    the specificity you get for it. Far more meaningful than a 0.5 cut-off.
    """
    y = _as_binary(y_true)
    s = np.asarray(y_score, dtype=float).ravel()
    for t in np.unique(s)[::-1]:
        if sensitivity(y, (s >= t).astype(int)) >= target:
            return float(t)
    return float(np.min(s))


def summary(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Every metric at once. ``y_prob`` must be P(class 1), not a hard label."""
    y = _as_binary(y_true)
    p = np.asarray(y_prob, dtype=float).ravel()
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion(y, yhat)
    return {
        "sensitivity": sensitivity(y, yhat),
        "specificity": specificity(y, yhat),
        "ppv": ppv(y, yhat),
        "npv": npv(y, yhat),
        "pr_auc": average_precision(y, p),
        "roc_auc": roc_auc(y, p),
        "balanced_accuracy": balanced_accuracy(y, yhat),
        "f1": f1(y, yhat),
        "mcc": mcc(y, yhat),
        "kappa": cohen_kappa(y, yhat),
        "brier": brier(y, p),
        "ece": expected_calibration_error(y, p),
        "accuracy": accuracy(y, yhat),
        "threshold": float(threshold),
        "n": int(y.size),
        "n_pos": int(y.sum()),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def aggregate(per_fold: list[dict[str, float]], keys: list[str] | None = None
              ) -> dict[str, dict[str, float]]:
    """Mean, sd and n across folds. Report mean +/- sd, never the best fold.

    ``sd`` uses ddof=1. It is not a confidence interval: folds share training data,
    so the sd understates true uncertainty. Use it to see whether a gap is smaller
    than fold-to-fold noise -- and if it is, do not claim the gap.
    """
    if not per_fold:
        raise ValueError("no folds to aggregate")
    keys = keys or [k for k in ALL_METRICS if k in per_fold[0]]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        v = np.array([f[k] for f in per_fold], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = {
            "mean": float(v.mean()) if v.size else float("nan"),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "n_folds": int(v.size),
            "min": float(v.min()) if v.size else float("nan"),
            "max": float(v.max()) if v.size else float("nan"),
        }
    return out


def bootstrap_ci(y_true, y_prob, metric: str = "sensitivity", threshold: float = 0.5,
                 n_boot: int = 2000, alpha: float = 0.05, seed: int = 20260830
                 ) -> tuple[float, float]:
    """Percentile bootstrap CI for one metric on one test set.

    Use this for the final locked holdout, where cross-validation sd is not
    available. With a few hundred test rows the interval is wide -- that width is
    the point, and showing it is more convincing than hiding it.
    """
    y = _as_binary(y_true)
    p = np.asarray(y_prob, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, y.size, y.size)
        if np.unique(y[i]).size < 2:
            continue
        vals.append(summary(y[i], p[i], threshold)[metric])
    if not vals:
        return float("nan"), float("nan")
    v = np.array(vals, dtype=float)
    return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))
