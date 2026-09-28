"""Significance tests, sized for this dataset.

Why this module exists: with roughly 180 test rows per fold, a 1-point accuracy
difference is about two patients. Reporting "quantum beat classical by 1.2%"
without a test is not a finding, it is rounding. The tests below answer the
question "could this gap be noise?" and the answer is usually yes.

* ``mcnemar`` compares two models on the SAME test set, using only the rows where
  they disagree. That pairing is what makes it powerful on small data.
* ``corrected_resampled_ttest`` compares two models across repeated CV folds. A
  plain paired t-test is badly over-confident there, because folds share training
  data; this applies the Nadeau-Bengio variance correction.
* ``paired_t_test`` is the plain version, for the case where the observations
  really are independent (one per dataset draw, folds already averaged inside).
* ``tost_equivalence`` answers the *opposite* question: not "is there a
  difference" but "how large a difference can we rule out". See below.

**On failing to reject.** A large p-value is not evidence of no effect; it is the
absence of evidence of one, and the two get conflated constantly in the QML
literature -- an underpowered comparison that finds nothing gets written up as
"comparable performance". The fix is an equivalence test. ``tost_equivalence``
runs two one-sided tests against a margin and reports the smallest margin the
data can actually support, so the claim becomes "any effect larger than X is
excluded at 95%" rather than "we saw nothing". Quote that bound, and quote it
next to the observed difference so a reader can see both.

**On the t distribution.** This module used to reach for scipy and fall back to a
normal approximation when it was absent, with a note admitting the fallback was
"slightly optimistic". That is survivable when you are trying to reject a null and
failing, but it is not survivable for an equivalence claim, where an
anti-conservative tail makes it *easier* to declare two things the same. So the
exact Student-t tail is computed here, from a regularized incomplete beta, with no
optional dependency and no approximation branch. It is checked against published
critical values in ``tests/test_stats.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["McNemarResult", "mcnemar", "corrected_resampled_ttest", "holm_bonferroni",
           "betainc_regularised", "student_t_sf", "student_t_ppf", "PairedTResult",
           "paired_t_test", "EquivalenceResult", "tost_equivalence"]


# ------------------------------------------------------- the t distribution
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method).

    Converges quickly for ``x < (a+1)/(a+b+2)``; the caller is responsible for
    applying the reflection ``I_x(a,b) = 1 - I_{1-x}(b,a)`` outside that range,
    where this series converges slowly or not at all.
    """
    maxit, eps, fpmin = 300, 3.0e-16, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < fpmin:
            d = fpmin
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < fpmin:
            d = fpmin
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise RuntimeError(
        f"incomplete beta did not converge at a={a}, b={b}, x={x}. Returning an "
        f"unconverged value would produce a plausible but wrong p-value, which is "
        f"the failure mode this whole module exists to prevent."
    )


def betainc_regularised(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)``, in pure Python.

    Only needed because scipy is not a dependency of this project. Verified against
    analytically known cases and against published t critical values.
    """
    if not (a > 0.0 and b > 0.0):
        raise ValueError(f"a and b must be positive, got a={a}, b={b}")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: float) -> float:
    """Upper tail ``P(T > t)`` for Student's t with ``df`` degrees of freedom.

    Uses ``P(T > |t|) = I_{df/(df+t^2)}(df/2, 1/2) / 2``. Note the two-sided p-value
    is then just ``I_x(df/2, 1/2)`` with no factor, which is worth knowing when
    reading the callers.
    """
    if df <= 0:
        raise ValueError(f"degrees of freedom must be positive, got {df}")
    t = float(t)
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    half = 0.5 * betainc_regularised(df / 2.0, 0.5, df / (df + t * t))
    return half if t >= 0.0 else 1.0 - half


def student_t_ppf(p: float, df: float) -> float:
    """Quantile ``t`` such that ``P(T <= t) = p``. Bisection on the tail.

    Bisection rather than a series because this is called a handful of times per
    run, so 60 evaluations of an already-fast function costs nothing, and a wrong
    quantile would silently mis-size every confidence interval in the results.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    if p == 0.5:
        return 0.0
    lo, hi = -1.0e6, 1.0e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if (1.0 - student_t_sf(mid, df)) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, abs(lo)):
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class McNemarResult:
    n_both_correct: int
    n_a_only: int        # a correct, b wrong
    n_b_only: int        # b correct, a wrong
    n_both_wrong: int
    statistic: float
    p_value: float
    method: str

    @property
    def n_discordant(self) -> int:
        return self.n_a_only + self.n_b_only

    def verdict(self, alpha: float = 0.05) -> str:
        if self.n_discordant < 10:
            return (f"inconclusive: only {self.n_discordant} discordant cases, too few "
                    f"to test. Do not claim a difference.")
        if self.p_value >= alpha:
            return (f"no significant difference (p={self.p_value:.3f}). Report the two "
                    f"models as indistinguishable on this data.")
        better = "A" if self.n_a_only > self.n_b_only else "B"
        return (f"{better} is better (p={self.p_value:.3f}), by "
                f"{abs(self.n_a_only - self.n_b_only)} patients out of "
                f"{self.n_discordant} discordant.")


def mcnemar(y_true, pred_a, pred_b, exact: bool | None = None) -> McNemarResult:
    """Paired test on two models' hard predictions over one test set.

    ``exact`` defaults to using the exact binomial when there are fewer than 25
    discordant pairs (where the chi-square approximation is unreliable) and the
    continuity-corrected chi-square otherwise.
    """
    y = np.asarray(y_true).ravel()
    a = (np.asarray(pred_a).ravel() == y)
    b = (np.asarray(pred_b).ravel() == y)
    if not (y.size == a.size == b.size):
        raise ValueError("y_true, pred_a and pred_b must be the same length")

    both = int(np.sum(a & b))
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    neither = int(np.sum(~a & ~b))
    n = a_only + b_only

    if n == 0:
        return McNemarResult(both, 0, 0, neither, 0.0, 1.0, "identical predictions")

    use_exact = (n < 25) if exact is None else exact
    if use_exact:
        k = min(a_only, b_only)
        tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
        p = min(1.0, 2.0 * tail)
        return McNemarResult(both, a_only, b_only, neither, float(k), float(p),
                             "exact binomial")

    stat = (abs(a_only - b_only) - 1) ** 2 / n           # Edwards continuity correction
    p = math.erfc(math.sqrt(stat / 2.0))                  # chi-square, 1 df, two-sided
    return McNemarResult(both, a_only, b_only, neither, float(stat), float(p),
                         "chi-square with continuity correction")


def corrected_resampled_ttest(scores_a, scores_b, n_train: int, n_test: int,
                              ) -> tuple[float, float, str]:
    """Nadeau-Bengio corrected paired t-test over repeated cross-validation scores.

    Returns ``(t_statistic, p_value, note)``. The correction inflates the variance
    by ``1 + n_test/n_train`` because resampled folds overlap in training data; the
    uncorrected test can report p < 0.001 for differences that vanish on new data.

    The t distribution is the exact one from ``student_t_sf``. An earlier version
    used scipy when present and a normal approximation otherwise, which meant the
    same fold scores produced different p-values on different machines.
    """
    a = np.asarray(scores_a, dtype=float).ravel()
    b = np.asarray(scores_b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("paired test needs equal-length score vectors")
    if a.size < 2:
        raise ValueError("need at least 2 folds")

    d = a - b
    k = d.size
    mean, var = float(d.mean()), float(d.var(ddof=1))
    if var <= 0:
        return 0.0, 1.0, "identical fold scores"

    var_corrected = var * (1.0 / k + n_test / max(n_train, 1))
    t = mean / math.sqrt(var_corrected)
    df = k - 1
    p = 2.0 * student_t_sf(abs(t), df)
    return float(t), float(p), f"Nadeau-Bengio corrected, exact t, df={df}"


@dataclass(frozen=True)
class PairedTResult:
    n: int
    df: int
    mean: float
    se: float
    t: float
    p_value: float
    ci_low: float
    ci_high: float
    alpha: float

    def verdict(self) -> str:
        if self.p_value < self.alpha:
            direction = "A > B" if self.mean > 0 else "A < B"
            return (f"{direction}: {self.mean:+.4f} "
                    f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}], p={self.p_value:.4f}")
        return (f"no detected difference: {self.mean:+.4f} "
                f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}], p={self.p_value:.3f}. "
                f"This is not evidence of equivalence -- run tost_equivalence.")


def paired_t_test(diffs, alpha: float = 0.05) -> PairedTResult:
    """Plain paired t over *independent* observations, with a confidence interval.

    Use this when each observation is a separate dataset draw and any within-draw
    resampling has already been averaged out. In that situation the observations
    are genuinely independent and the Nadeau-Bengio correction does not apply --
    it corrects for training-set overlap between folds, and there is none between
    draws. Applying it anyway would be conservative rather than wrong, but it would
    also throw away the power that generating fresh data was meant to buy.

    Feeding this per-fold scores from one dataset instead is the classic mistake:
    folds sharing ~80% of their training rows are not independent, and the
    resulting t is inflated.
    """
    d = np.asarray(diffs, dtype=float).ravel()
    n = d.size
    if n < 2:
        raise ValueError("need at least 2 paired observations")
    df = n - 1
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    if se <= 0.0:
        return PairedTResult(n, df, mean, 0.0, float("nan"), 1.0, mean, mean, alpha)
    t = mean / se
    p = 2.0 * student_t_sf(abs(t), df)
    crit = student_t_ppf(1.0 - alpha / 2.0, df)
    return PairedTResult(n, df, mean, se, float(t), float(p),
                         mean - crit * se, mean + crit * se, alpha)


@dataclass(frozen=True)
class EquivalenceResult:
    n: int
    df: int
    mean: float
    se: float
    margin: float
    p_value: float
    equivalent: bool
    ci_low: float
    ci_high: float
    bound: float
    alpha: float

    def verdict(self) -> str:
        if self.equivalent:
            return (f"equivalent within +-{self.margin:.4f} (p={self.p_value:.4f}); "
                    f"effects larger than {self.bound:.4f} are excluded")
        return (f"NOT equivalent within +-{self.margin:.4f} (p={self.p_value:.3f}); "
                f"the data only exclude effects larger than {self.bound:.4f}")


def tost_equivalence(diffs, margin: float, alpha: float = 0.05) -> EquivalenceResult:
    """Two one-sided tests: can we rule out a difference larger than ``margin``?

    This is the test that a null result actually needs. The usual paired t asks
    "is the difference non-zero" and answers "we cannot tell"; TOST asks "is the
    difference smaller than something we would care about" and can answer yes.

    Equivalence at ``margin`` holds exactly when the two-sided ``1 - 2*alpha``
    confidence interval lies entirely inside ``(-margin, +margin)``. That is why
    the interval reported here is the 90% one at the default alpha=0.05 and not
    the 95% one -- pairing a 95% interval with a 5% TOST is a common and
    conservative-looking error that actually mis-states the test being run.

    ``bound`` is the smallest margin at which equivalence *would* be declared,
    i.e. the larger absolute end of that interval. It is the useful number to
    report, because it does not depend on choosing a margin in advance: "effects
    larger than ``bound`` are excluded" is a claim the data support on their own.
    """
    if margin <= 0:
        raise ValueError("equivalence margin must be positive")
    d = np.asarray(diffs, dtype=float).ravel()
    n = d.size
    if n < 2:
        raise ValueError("need at least 2 paired observations")
    df = n - 1
    mean = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    if se <= 0.0:
        eq = abs(mean) < margin
        return EquivalenceResult(n, df, mean, 0.0, margin, 0.0 if eq else 1.0, eq,
                                 mean, mean, abs(mean), alpha)
    # H0_lower: mu <= -margin, rejected by a large positive t_lower
    # H0_upper: mu >= +margin, rejected by a large negative t_upper
    p_lower = student_t_sf((mean + margin) / se, df)
    p_upper = 1.0 - student_t_sf((mean - margin) / se, df)
    p = max(p_lower, p_upper)
    crit = student_t_ppf(1.0 - alpha, df)              # one-sided -> 1-2a interval
    lo, hi = mean - crit * se, mean + crit * se
    return EquivalenceResult(n, df, mean, se, margin, float(p), bool(p < alpha),
                             lo, hi, max(abs(lo), abs(hi)), alpha)


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05
                    ) -> dict[str, tuple[float, bool]]:
    """Step-down multiple-comparison correction.

    Comparing one quantum model against five classical baselines is five tests; at
    alpha=0.05 there is a ~23% chance one looks significant by luck. Apply this
    before claiming any single comparison, and say in the paper that you did.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, tuple[float, bool]] = {}
    still_rejecting = True
    for i, (name, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        still_rejecting = still_rejecting and adj < alpha
        out[name] = (float(adj), bool(still_rejecting))
    return out
