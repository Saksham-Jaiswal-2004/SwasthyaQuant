"""The statistics, checked against things that are not the statistics.

``eval/stats.py`` decides whether every comparison in this project counts as a
finding, and until now it had no tests at all. It also contains a hand-rolled
incomplete beta function, which is precisely the kind of code that returns a
plausible wrong number rather than failing -- an unconverged continued fraction
does not raise, it just gives you p = 0.03 instead of p = 0.07.

So nothing here compares the module to itself. The references are:

* **Closed forms.** ``I_x(1,1) = x``, ``I_x(2,1) = x^2``, ``I_x(1,2) = 2x - x^2``
  and ``I_x(1/2,1/2) = (2/pi) arcsin(sqrt x)`` are exact. So are the t tails at
  1 and 2 degrees of freedom: ``P(T>t) = 1/2 - arctan(t)/pi`` for the Cauchy, and
  ``1/2 (1 - t/sqrt(t^2+2))`` at df=2. Those four beta cases and two t cases pin
  the implementation analytically.
* **Published critical values.** The two-sided 5% points of the t distribution
  from any standard table must return p = 0.05.
* **Internal cross-checks between two independently computed routes.** TOST
  computes a p-value from two one-sided tails and a bound from a confidence
  interval; the equivalence decision must agree either way. If it does not, one
  of the two is wrong.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from qheart.eval.stats import (betainc_regularised, corrected_resampled_ttest,
                               holm_bonferroni, mcnemar, paired_t_test,
                               student_t_ppf, student_t_sf, tost_equivalence)

# Two-sided 5% critical values, standard tables. If student_t_sf is right, each of
# these must map back to p = 0.05.
CRIT_05 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 10: 2.228,
           19: 2.093, 20: 2.086, 30: 2.042, 60: 2.000, 120: 1.980}


# ------------------------------------------------------------ incomplete beta
@pytest.mark.parametrize("x", [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
def test_betainc_matches_closed_forms(x):
    """Four cases where the incomplete beta is elementary."""
    assert betainc_regularised(1.0, 1.0, x) == pytest.approx(x, abs=1e-12)
    assert betainc_regularised(2.0, 1.0, x) == pytest.approx(x * x, abs=1e-12)
    assert betainc_regularised(1.0, 2.0, x) == pytest.approx(2 * x - x * x, abs=1e-12)
    assert betainc_regularised(0.5, 0.5, x) == pytest.approx(
        (2.0 / math.pi) * math.asin(math.sqrt(x)), abs=1e-12)


def test_betainc_is_monotone_and_bounded():
    """Both branches of the reflection are exercised here; a discontinuity at the
    switchover ``x = (a+1)/(a+b+2)`` would show up as a monotonicity failure."""
    vals = [betainc_regularised(3.5, 2.5, x) for x in np.linspace(1e-6, 1 - 1e-6, 400)]
    assert all(b >= a - 1e-15 for a, b in zip(vals, vals[1:]))
    assert min(vals) > 0.0 and max(vals) < 1.0
    assert betainc_regularised(3.5, 2.5, 0.0) == 0.0
    assert betainc_regularised(3.5, 2.5, 1.0) == 1.0


def test_betainc_reflection_is_consistent():
    """I_x(a,b) = 1 - I_{1-x}(b,a). The two sides take different code branches."""
    for a, b, x in [(2.0, 7.0, 0.2), (7.0, 2.0, 0.8), (0.5, 4.0, 0.6), (10.0, 10.0, 0.45)]:
        assert betainc_regularised(a, b, x) == pytest.approx(
            1.0 - betainc_regularised(b, a, 1.0 - x), abs=1e-13)


def test_betainc_rejects_bad_parameters():
    with pytest.raises(ValueError):
        betainc_regularised(0.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        betainc_regularised(1.0, -1.0, 0.5)


# --------------------------------------------------------- the t distribution
def test_t_tail_matches_cauchy_closed_form():
    """df=1 is the Cauchy distribution, whose tail is exact."""
    for t in [-4.0, -1.0, -0.3, 0.0, 0.3, 1.0, 4.0, 30.0]:
        assert student_t_sf(t, 1) == pytest.approx(0.5 - math.atan(t) / math.pi,
                                                   abs=1e-12)


def test_t_tail_matches_df2_closed_form():
    """At df=2 the tail is 1/2 (1 - t/sqrt(t^2+2)), also exact."""
    for t in [-3.0, -0.5, 0.0, 0.5, 3.0, 12.0]:
        want = 0.5 * (1.0 - t / math.sqrt(t * t + 2.0))
        assert student_t_sf(t, 2) == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize("df", sorted(CRIT_05))
def test_published_critical_values_give_five_percent(df):
    """The headline check. A table value is rounded to three decimals, which moves
    the p-value by well under 1e-4, so 3e-4 is a loose tolerance rather than a
    generous one."""
    p = 2.0 * student_t_sf(CRIT_05[df], df)
    assert p == pytest.approx(0.05, abs=3e-4), f"df={df}: p={p:.6f}"


def test_t_tail_symmetry_and_centre():
    for df in (1, 3, 19, 100):
        assert student_t_sf(0.0, df) == pytest.approx(0.5, abs=1e-12)
        for t in (0.4, 1.3, 2.9):
            assert student_t_sf(-t, df) == pytest.approx(1.0 - student_t_sf(t, df),
                                                         abs=1e-13)


def test_t_approaches_normal_at_large_df():
    """t_{0.975} -> 1.959964. Guards against a df scaling error that would be
    invisible at the small df this project usually runs at."""
    assert student_t_ppf(0.975, 100000) == pytest.approx(1.959964, abs=1e-4)


def test_ppf_inverts_sf():
    for df in (1, 2, 3, 7, 19, 40):
        for p in (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999):
            t = student_t_ppf(p, df)
            assert (1.0 - student_t_sf(t, df)) == pytest.approx(p, abs=1e-9)


@pytest.mark.parametrize("df", sorted(CRIT_05))
def test_ppf_reproduces_the_tables(df):
    assert student_t_ppf(0.975, df) == pytest.approx(CRIT_05[df], abs=1e-3)


def test_t_rejects_bad_df_and_probability():
    with pytest.raises(ValueError):
        student_t_sf(1.0, 0)
    with pytest.raises(ValueError):
        student_t_ppf(0.0, 5)
    with pytest.raises(ValueError):
        student_t_ppf(1.0, 5)


# ------------------------------------------------------------- paired t test
def test_paired_t_matches_hand_computation():
    """d = 1..5: mean 3, sd sqrt(2.5), se sqrt(0.5), t = 3/sqrt(0.5) = 4.2426.
    The p-value is bracketed by the df=4 table entries at 0.02 (3.747) and
    0.01 (4.604), so it must land strictly between."""
    r = paired_t_test([1.0, 2.0, 3.0, 4.0, 5.0])
    assert r.n == 5 and r.df == 4
    assert r.mean == pytest.approx(3.0, abs=1e-12)
    assert r.se == pytest.approx(math.sqrt(0.5), abs=1e-12)
    assert r.t == pytest.approx(3.0 / math.sqrt(0.5), abs=1e-12)
    assert 0.01 < r.p_value < 0.02


def test_paired_t_interval_brackets_the_mean_and_matches_the_p_value():
    """A CI that excludes zero and a p-value above alpha are contradictory. This
    catches a mismatched critical value, which is otherwise silent."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        d = rng.normal(0.02, 0.05, size=int(rng.integers(3, 25)))
        r = paired_t_test(d)
        assert r.ci_low < r.mean < r.ci_high
        assert (r.p_value < 0.05) == (r.ci_low > 0.0 or r.ci_high < 0.0)


def test_paired_t_needs_two_observations():
    with pytest.raises(ValueError):
        paired_t_test([0.3])


def test_paired_t_survives_identical_observations():
    """Zero variance must not divide by zero. It is also not a significant result,
    which is the non-obvious half: a constant nonzero difference has no estimable
    standard error, so nothing can be claimed from it."""
    r = paired_t_test([0.5, 0.5, 0.5, 0.5])
    assert r.p_value == 1.0 and r.se == 0.0


# --------------------------------------------------------------------- TOST
def test_tost_bound_agrees_with_the_p_value_route():
    """``bound`` comes from a confidence interval and ``p_value`` from two
    one-sided tails, computed independently. Equivalence must be declared at any
    margin above the bound and refused at any margin below it -- if these two
    disagree, one of them is wrong."""
    rng = np.random.default_rng(1)
    for _ in range(40):
        d = rng.normal(0.001, 0.02, size=int(rng.integers(4, 30)))
        b = tost_equivalence(d, margin=1.0).bound
        assert tost_equivalence(d, margin=b * 1.02).equivalent
        assert not tost_equivalence(d, margin=b * 0.98).equivalent


def test_tost_declares_equivalence_for_a_tight_null():
    """The whole point: a difference this small and this well measured should be
    positively declared negligible, not merely 'not significant'."""
    rng = np.random.default_rng(2)
    d = rng.normal(0.0005, 0.004, size=40)
    r = tost_equivalence(d, margin=0.01)
    assert r.equivalent and r.p_value < 0.05
    assert r.bound < 0.01


def test_tost_refuses_equivalence_when_underpowered():
    """Four noisy observations must NOT license an equivalence claim, even though
    the plain t-test also fails to reject. This is the exact confusion the test
    exists to prevent, and it is the reason the seed count was raised."""
    d = np.array([0.002, -0.05, 0.04, -0.03])
    assert paired_t_test(d).p_value > 0.05          # no detected difference
    assert not tost_equivalence(d, margin=0.01).equivalent    # but not equivalent
    assert tost_equivalence(d, margin=0.01).bound > 0.01


def test_tost_refuses_equivalence_for_a_real_difference():
    rng = np.random.default_rng(3)
    d = rng.normal(0.08, 0.01, size=20)
    assert not tost_equivalence(d, margin=0.01).equivalent


def test_tost_interval_is_the_one_minus_two_alpha_interval():
    """At alpha=0.05 TOST corresponds to a 90% interval, not a 95% one. Using the
    95% interval would make the reported bound wider than the test actually
    requires and quietly misdescribe the procedure."""
    d = np.random.default_rng(4).normal(0.0, 0.02, size=25)
    r = tost_equivalence(d, margin=0.05, alpha=0.05)
    crit90 = student_t_ppf(0.95, r.df)
    assert r.ci_high == pytest.approx(r.mean + crit90 * r.se, abs=1e-12)
    assert r.ci_high < paired_t_test(d).ci_high      # strictly tighter than 95%


def test_tost_rejects_a_nonpositive_margin():
    with pytest.raises(ValueError):
        tost_equivalence([0.1, 0.2, 0.3], margin=0.0)


# --------------------------------------------------- the pre-existing helpers
def test_nadeau_bengio_is_always_more_conservative_than_a_plain_t():
    """The correction exists to shrink |t|. If a refactor ever inverted the
    variance expression the test would still run and would report more
    significance, not less."""
    rng = np.random.default_rng(5)
    a = rng.normal(0.8, 0.05, size=10)
    b = a - rng.normal(0.02, 0.01, size=10)
    t_corr, p_corr, note = corrected_resampled_ttest(a, b, n_train=800, n_test=200)
    t_plain = paired_t_test(a - b).t
    assert abs(t_corr) < abs(t_plain)
    assert p_corr > paired_t_test(a - b).p_value
    assert "exact t" in note and "scipy" not in note


def test_mcnemar_exact_matches_hand_computed_binomial():
    """5 discordant pairs all favouring A: two-sided p = 2 * 0.5^5 = 0.0625."""
    y = np.zeros(20, dtype=int)
    a = np.zeros(20, dtype=int)          # A correct everywhere
    b = np.zeros(20, dtype=int)
    b[:5] = 1                            # B wrong on 5 rows, right on the rest
    r = mcnemar(y, a, b)
    assert (r.n_a_only, r.n_b_only) == (5, 0)
    assert r.p_value == pytest.approx(0.0625, abs=1e-12)
    assert r.method == "exact binomial"


def test_mcnemar_calls_a_tie_a_tie():
    y = np.zeros(10, dtype=int)
    r = mcnemar(y, y.copy(), y.copy())
    assert r.p_value == 1.0 and r.n_discordant == 0
    assert "inconclusive" in r.verdict()


def test_mcnemar_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mcnemar(np.zeros(5), np.zeros(5), np.zeros(4))


def test_holm_bonferroni_steps_down_and_stops():
    """Three tests: 0.001 survives (x3), 0.03 fails (x2 = 0.06), and everything
    after it must be marked non-rejecting regardless of its own adjusted value."""
    out = holm_bonferroni({"a": 0.001, "b": 0.03, "c": 0.04})
    assert out["a"][0] == pytest.approx(0.003) and out["a"][1]
    assert out["b"][0] == pytest.approx(0.06) and not out["b"][1]
    assert not out["c"][1]
