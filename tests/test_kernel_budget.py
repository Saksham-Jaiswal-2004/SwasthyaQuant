"""The O(N^2) guard. This is the test that prevents someone discovering the cost wall by
watching a progress bar for two days in the last week before submission.
"""

from __future__ import annotations

import pytest

from qheart.quantum.budget import estimate_kernel_cost, fmt_seconds, guard_kernel_size
from qheart.quantum.encodings import check_range, n_qubits_for


def test_upper_triangle_only():
    # n(n-1)/2, not n^2 -- the kernel is symmetric with a unit diagonal, so half the
    # matrix is free. 918 rows: 420,903 evaluations rather than 842,724.
    e = estimate_kernel_cost(918)
    assert e.train_evals == 420_903 == 918 * 917 // 2
    assert e.train_evals < 918 ** 2


def test_test_block_is_rectangular_not_triangular():
    # Cross terms between test and train have no symmetry to exploit: n_test x n_train.
    e = estimate_kernel_cost(700, 200)
    assert e.test_evals == 200 * 700
    assert e.total_evals == e.train_evals + e.test_evals


def test_cost_is_quadratic_not_linear():
    a = estimate_kernel_cost(1000).total_evals
    b = estimate_kernel_cost(2000).total_evals
    assert 3.9 < b / a < 4.1, "doubling N must roughly quadruple cost"


def test_the_number_that_justifies_918_rows():
    small = estimate_kernel_cost(918).total_evals
    big = estimate_kernel_cost(20_000).total_evals
    assert small < 5e5
    assert big > 1e8
    assert big / small > 400        # this ratio is the answer to "why not more data?"


def test_guard_allows_the_project_size_and_refuses_a_hopeless_one():
    guard_kernel_size(918, max_samples=2000)            # must not raise
    with pytest.raises(ValueError, match="circuit evaluations"):
        guard_kernel_size(50_000, max_samples=2000)


def test_guard_can_be_downgraded_to_a_warning_but_not_by_default():
    guard_kernel_size(50_000, max_samples=2000, hard=False)   # prints, does not raise


def test_fmt_seconds_reads_sensibly():
    assert fmt_seconds(30).endswith("s")
    assert "min" in fmt_seconds(600)
    assert "h" in fmt_seconds(20_000)
    assert "days" in fmt_seconds(500_000)


def test_dense_encoding_halves_the_qubit_count():
    assert n_qubits_for(8, "angle") == 8
    assert n_qubits_for(8, "dense_angle") == 4      # why the project uses 4 qubits
    assert n_qubits_for(8, "amplitude") == 3
    assert n_qubits_for(7, "dense_angle") == 4      # odd counts round up, not down


def test_unknown_encoding_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown encoding"):
        n_qubits_for(8, "basis")


def test_angle_range_check_rejects_wrapped_values():
    import numpy as np
    check_range(np.array([0.0, 1.0, np.pi]))
    # A z-scored feature at +4 sigma rotates past 2*pi and wraps: an extreme patient
    # silently reads as an average one. This must be an error, not a clip.
    with pytest.raises(ValueError, match="angle encoding needs"):
        check_range(np.array([0.0, 7.5]))
    with pytest.raises(ValueError, match="NaN or inf"):
        check_range(np.array([0.0, np.nan]))
