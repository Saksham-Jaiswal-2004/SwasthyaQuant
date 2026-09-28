"""Metrics checked against values computed by hand, not against another library.

If these ever fail after a refactor, trust the test: the expected numbers below
were derived on paper and are re-derivable from the comments.
"""

import numpy as np
import pytest

from qheart.eval import metrics as M


def test_confusion_order_and_counts():
    # y   1 1 1 1 0 0 0 0
    # p .9 .8 .4 .3 .7 .6 .2 .1   at 0.5 -> yhat 1 1 0 0 1 1 0 0
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.4, 0.3, 0.7, 0.6, 0.2, 0.1]
    yhat = (np.array(p) >= 0.5).astype(int)
    assert M.confusion(y, yhat) == (2, 2, 2, 2)      # tn, fp, fn, tp
    assert M.sensitivity(y, yhat) == pytest.approx(0.5)
    assert M.specificity(y, yhat) == pytest.approx(0.5)
    assert M.ppv(y, yhat) == pytest.approx(0.5)
    assert M.npv(y, yhat) == pytest.approx(0.5)
    assert M.accuracy(y, yhat) == pytest.approx(0.5)
    # tp*tn - fp*fn = 4 - 4 = 0
    assert M.mcc(y, yhat) == pytest.approx(0.0)
    # po = 0.5, pe = (4*4 + 4*4)/64 = 0.5 -> kappa = 0
    assert M.cohen_kappa(y, yhat) == pytest.approx(0.0)
    assert M.youden_j(y, yhat) == pytest.approx(0.0)


def test_perfect_and_inverted():
    y = [0, 0, 1, 1]
    good = [0.1, 0.2, 0.8, 0.9]
    assert M.roc_auc(y, good) == pytest.approx(1.0)
    assert M.average_precision(y, good) == pytest.approx(1.0)
    assert M.summary(y, good)["sensitivity"] == pytest.approx(1.0)
    # Perfectly wrong ranking -> AUC 0, which is information, not a bug.
    assert M.roc_auc(y, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_roc_auc_concordant_pairs():
    # pos {0.9, 0.6}, neg {0.8, 0.1}: 0.9>0.8, 0.9>0.1, 0.6<0.8, 0.6>0.1 -> 3/4
    assert M.roc_auc([1, 1, 0, 0], [0.9, 0.6, 0.8, 0.1]) == pytest.approx(0.75)


def test_roc_auc_all_ties_is_half():
    # Every score identical: no ranking information at all.
    assert M.roc_auc([1, 0, 1, 0], [0.5] * 4) == pytest.approx(0.5)
    # Half credit for a tied pair: pos {0.5}, neg {0.5, 0.1} -> (0.5 + 1)/2
    assert M.roc_auc([1, 0, 0], [0.5, 0.5, 0.1]) == pytest.approx(0.75)


def test_average_precision_matches_hand_calculation():
    # desc: (0.8,1) (0.4,0) (0.35,1) (0.1,0), n_pos = 2
    #   k=1 P=1    R=0.5
    #   k=2 P=0.5  R=0.5
    #   k=3 P=2/3  R=1.0
    #   k=4 P=0.5  R=1.0
    # AP = 0.5*1 + 0*0.5 + 0.5*(2/3) + 0*0.5 = 5/6
    ap = M.average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert ap == pytest.approx(5 / 6)


def test_average_precision_baseline_is_prevalence():
    # Random scores on a 25% prevalence problem sit near 0.25, not 0.5.
    rng = np.random.default_rng(0)
    y = np.repeat([1, 0], [250, 750])
    ap = M.average_precision(y, rng.random(1000))
    assert 0.18 < ap < 0.34


def test_brier_and_ece():
    assert M.brier([1, 0], [0.8, 0.3]) == pytest.approx((0.04 + 0.09) / 2)
    # Perfectly calibrated, perfectly confident -> zero error.
    assert M.expected_calibration_error([1, 1, 0, 0], [1.0, 1.0, 0.0, 0.0]) == pytest.approx(0.0)
    # Confidently wrong -> ECE 1.
    assert M.expected_calibration_error([0, 0], [1.0, 1.0]) == pytest.approx(1.0)


def test_threshold_at_sensitivity_trades_specificity():
    y = np.repeat([1, 0], [50, 50])
    rng = np.random.default_rng(1)
    s = np.r_[rng.normal(0.65, 0.15, 50), rng.normal(0.35, 0.15, 50)].clip(0, 1)
    t = M.threshold_at_sensitivity(y, s, target=0.90)
    assert M.sensitivity(y, (s >= t).astype(int)) >= 0.90
    # Buying sensitivity must cost specificity relative to the default cut-off.
    assert M.specificity(y, (s >= t).astype(int)) <= M.specificity(y, (s >= 0.5).astype(int))


def test_summary_keys_and_headline_present():
    s = M.summary([0, 1, 0, 1], [0.2, 0.7, 0.4, 0.9])
    for k in M.HEADLINE:
        assert k in s, "headline metrics must never be droppable"
    assert set(M.ALL_METRICS).issubset(s)
    assert s["tp"] + s["fn"] == s["n_pos"]


def test_aggregate_reports_spread_not_best():
    folds = [{"sensitivity": v} for v in (0.80, 0.90, 0.70)]
    a = M.aggregate(folds, keys=["sensitivity"])["sensitivity"]
    assert a["mean"] == pytest.approx(0.80)
    assert a["sd"] == pytest.approx(0.1)
    assert a["n_folds"] == 3 and a["max"] == pytest.approx(0.90)


def test_labels_must_be_binary():
    with pytest.raises(ValueError):
        M.confusion([0, 1, 2], [0, 1, 1])
