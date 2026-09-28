"""Parameter accounting. The headline claim of the whole project lives in these numbers,
so they get the most explicit tests in the repo.
"""

from __future__ import annotations

import pytest

from qheart.params import (assert_matched, bytes_at, count_circuit_params,
                           size_control_reducer)


def test_the_headline_number():
    # 4 qubits x 6 layers x 4 rotations (RY, RX, RZ, RY) = 96 angles = 384 bytes at fp32.
    c = count_circuit_params(4, 6)
    assert c.trainable == 96
    assert c.bytes_fp32() == 384


def test_entanglers_are_counted_separately_and_never_summed_in():
    # Paper P1's error: counting CNOTs inflates 16 angles/layer to 19, so 96 -> 114.
    c = count_circuit_params(4, 6)
    assert c.entanglers == 24
    assert c.trainable == 96            # unchanged by the entangler count
    assert c.trainable + c.entanglers == 120   # what you get if you make P1's mistake


def test_depth_four_is_sixtyfour():
    assert count_circuit_params(4, 4).trainable == 64


def test_count_scales_linearly_in_depth_and_qubits():
    a = count_circuit_params(4, 6).trainable
    assert count_circuit_params(4, 12).trainable == 2 * a
    assert count_circuit_params(8, 6).trainable == 2 * a


def test_invalid_geometry_raises():
    with pytest.raises(ValueError):
        count_circuit_params(0, 6)
    with pytest.raises(ValueError):
        count_circuit_params(4, 0)


def test_control_sizing_brackets_ninetysix():
    # params(h) = h*(d_in + 1 + d_out) + d_out = 13h + 4 for d_in=8, d_out=4.
    # 13*7 + 4 = 95 and 13*8 + 4 = 108, so 96 falls between them.
    s = size_control_reducer(96, d_in=8, d_out=4)
    assert (s.hidden_below, s.params_below) == (7, 95)
    assert (s.hidden_above, s.params_above) == (8, 108)
    assert not s.exact
    assert s.params_below <= 96 < s.params_above


def test_control_sizing_reports_exact_when_it_lands_exactly():
    # 13h + 4 = 95 at h = 7, so asking for 95 is an exact match.
    s = size_control_reducer(95, d_in=8, d_out=4)
    assert s.exact and s.params_below == 95


def test_control_sizing_never_returns_zero_width():
    s = size_control_reducer(5, d_in=8, d_out=4)
    assert s.hidden_below >= 1


def test_bytes_at_precision():
    assert bytes_at(96, "fp32") == 384
    assert bytes_at(96, "fp16") == 192
    assert bytes_at(96, "fp64") == 768
    with pytest.raises(ValueError):
        bytes_at(96, "int4")


def test_assert_matched_accepts_the_bracketing_control():
    assert_matched(96, 95, tolerance=0.05)          # ~1% off, fine


def test_assert_matched_rejects_a_drifted_control():
    # The failure this exists to catch: the depth changed, the control did not, and the
    # table still says "parameter-matched".
    with pytest.raises(AssertionError, match="stop calling it matched"):
        assert_matched(96, 512, tolerance=0.05)


def test_assert_matched_rejects_nonsense_input():
    with pytest.raises(ValueError):
        assert_matched(0, 95)
