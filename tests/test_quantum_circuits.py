"""Every circuit in this repo, checked against something that is not itself.

The rule these tests enforce: a quantum claim is only as good as the thing it was
checked against. Comparing a simulator to itself proves nothing, so each test here has
an *independent* reference -- a dense matrix exponential, a finite difference, a
brute-force construction, or an analytically known value.

The history behind several of them is worth keeping, because each marks a bug that
passed inspection and was only caught by execution:

* ``test_ryz_matches_dense`` -- the Y-Z-string rotation was implemented with the
  conjugation applied in the wrong order. It ran, stayed unitary, preserved the norm,
  and rotated the wrong way.
* ``test_agp_circuit_matches_operator`` -- the counterdiabatic term was originally a
  scalar-weighted single-qubit RY. Plausible, self-consistent, and wrong: the real
  commutator produces multi-body Y (x) Z strings.
* ``test_dcqf_is_not_silently_dead`` -- with the schedule derivative evaluated at t=T
  the AGP vanishes identically, every feature is exactly 0.0, and nothing raises.
* ``test_parameter_shift_matches_finite_difference`` -- an analytic gradient that is
  subtly wrong trains to a worse optimum instead of failing.
"""
from __future__ import annotations

import numpy as np
import pytest

from qheart.models.algorithms import QSVC, HybridQNN, VQC
from qheart.models.dcqf_control import ChainProductControl, ScrambledDCQF
from qheart.models.heads import KernelLogisticHead, LogisticHead, sigmoid
from qheart.quantum.ansatz import weight_shape
from qheart.quantum.circuits import (fidelity, fidelity_compute_uncompute,
                                     gram_matrix, n_weights, prepare)
from qheart.quantum.dcqf import (DCQFExtractor, agp_alpha, schedule_derivative)
from qheart.quantum.statevector import State

I2 = np.eye(2, dtype=complex)
X_ = np.array([[0, 1], [1, 0]], dtype=complex)
Y_ = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_ = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_op(n, ops):
    """Dense operator with ``ops`` placed on the given qubits. Qubit 0 is the most
    significant bit, i.e. leftmost in the Kronecker product -- the same convention
    ``State`` uses. If these two disagree, every correlation observable pairs the
    wrong variables, so this helper is deliberately written out rather than imported."""
    out = np.array([[1.0 + 0j]])
    for q in range(n):
        out = np.kron(out, ops.get(q, I2))
    return out


def expm_pauli(P, coef):
    """exp(-i coef P) for a Pauli string, using P^2 = I."""
    return np.cos(coef) * np.eye(P.shape[0], dtype=complex) - 1j * np.sin(coef) * P


# --------------------------------------------------------------- statevector
def test_ryz_matches_dense():
    rng = np.random.default_rng(7)
    worst = 0.0
    for n in (2, 3, 4, 5):
        for _ in range(10):
            yq = int(rng.integers(0, n))
            others = [q for q in range(n) if q != yq]
            zq = list(rng.permutation(others)[:int(rng.integers(0, len(others) + 1))])
            theta = float(rng.uniform(-3.0, 3.0))
            ops = {yq: Y_}
            for q in zq:
                ops[q] = Z_
            U = expm_pauli(kron_op(n, ops), theta / 2.0)
            v0 = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
            v0 /= np.linalg.norm(v0)
            st = State(n, v0.copy()).ryz_string(yq, zq, theta)
            worst = max(worst, float(np.max(np.abs(st.vec - U @ v0))))
    assert worst < 1e-12, f"ryz_string != exp(-i th/2 Y(x)Z...Z); worst {worst:.2e}"


def test_ryz_sign_is_plus_y():
    """Guards the specific failure that a flipped conjugation produces: the gate is
    still a valid Y rotation, just the wrong direction."""
    th = 0.7
    st = State(1).ryz_string(0, [], th)
    assert np.allclose(st.vec, [np.cos(th / 2), np.sin(th / 2)])
    assert np.allclose(State(1).ry(0, th).vec, st.vec)


def test_z_string_expectations_match_dense():
    rng = np.random.default_rng(11)
    n = 4
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    v /= np.linalg.norm(v)
    st = State(n, v.copy())
    for s in [(0,), (2,), (0, 1), (1, 3), (0, 1, 2), (0, 1, 2, 3)]:
        want = float(np.real(np.vdot(v, kron_op(n, {q: Z_ for q in s}) @ v)))
        assert abs(st.z_expectation(list(s)) - want) < 1e-12


# ---------------------------------------------------------------------- DCQF
@pytest.mark.parametrize("orders", [(2,), (3,), (2, 3)])
def test_agp_circuit_matches_operator(orders):
    """The executed circuit must equal the Hamiltonian written in the docstring,
    rebuilt here from dense operators with no reference to the gate code."""
    rng = np.random.default_rng(3)
    n = 4
    Xtr = rng.normal(size=(150, n))
    ex = DCQFExtractor(orders_encoded=orders, orders_read=(1, 2, 3), dt=1.0).fit(Xtr)
    x = rng.normal(size=n)

    psi = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)          # |+>^n
    for i, xi in enumerate(x):
        psi = expm_pauli(kron_op(n, {i: Z_}), float(xi)) @ psi
    for S, c in zip(ex.subsets_, ex.couplings_):
        psi = expm_pauli(kron_op(n, {i: Z_ for i in S}), float(c)) @ psi
    g = -2.0 * agp_alpha(x, ex.couplings_) * schedule_derivative(0.5, 1.0)
    for j, xj in enumerate(x):
        psi = expm_pauli(kron_op(n, {j: Y_}), g * float(xj)) @ psi
    for S, c in zip(ex.subsets_, ex.couplings_):
        for j in S:
            d = {i: Z_ for i in S if i != j}
            d[j] = Y_
            psi = expm_pauli(kron_op(n, d), g * float(c)) @ psi

    p = np.abs(psi) ** 2
    idx = np.arange(2 ** n)
    want = []
    for s in ex.observables_:
        par = np.zeros(2 ** n, dtype=np.int64)
        for q in s:
            par ^= (idx >> (n - 1 - q)) & 1
        want.append(float(np.dot(p, 1.0 - 2.0 * par)))
    got = ex.transform(x[None, :])[0]
    assert np.max(np.abs(got - np.array(want))) < 1e-12


def test_dcqf_is_not_silently_dead(monkeypatch):
    """With alpha forced to zero the circuit is diagonal, |+>^n stays uniform, and
    EVERY feature is exactly 0.0 -- with no exception raised anywhere. This test is
    the tripwire for that whole class of bug, including the lambda'(T) = 0 trap."""
    rng = np.random.default_rng(5)
    Xtr = rng.uniform(0, np.pi, size=(120, 5))
    ex = DCQFExtractor().fit(Xtr)
    live = ex.transform(Xtr[:30])
    assert np.max(np.abs(live)) > 0.05, "AGP too weak: extractor is effectively linear"

    import qheart.quantum.dcqf as mod
    monkeypatch.setattr(mod, "agp_alpha", lambda f, c: 0.0)
    dead = ex.transform(Xtr[:30])
    assert np.max(np.abs(dead)) < 1e-14


def test_schedule_derivative_vanishes_at_endpoints():
    """Documents the trap rather than trusting anyone to remember it."""
    assert schedule_derivative(1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert schedule_derivative(1e-12, 1.0) == pytest.approx(0.0, abs=1e-9)
    assert schedule_derivative(0.5, 1.0) > 1.5          # midpoint is where the signal is


def test_agp_alpha_is_intensive():
    """alpha must not decay with feature count. When it did (a SUM in the denominator
    rather than a mean) the kick fell from 8.9 degrees at 4 features to 1.5 at 22,
    and the extractor quietly linearised as the problem got bigger."""
    rng = np.random.default_rng(0)
    angles = []
    for n in (4, 8, 16):
        X = rng.uniform(0, np.pi, size=(80, n))
        d = DCQFExtractor(orders_encoded=(2,)).fit(X).diagnostics(X)
        angles.append(d["agp_angle_deg"])
    assert min(angles) > 15.0, f"AGP kick too small: {angles}"
    assert max(angles) / min(angles) < 1.5, f"AGP kick depends on n: {angles}"


def test_dcqf_couplings_are_fold_local():
    """Fitting on different data must produce different couplings. If it does not, the
    MI estimate is not reaching the encoding and the 'quantum' features are a fixed
    map -- which would make the leak-safety argument vacuous as well."""
    rng = np.random.default_rng(1)
    a = DCQFExtractor().fit(rng.uniform(0, np.pi, size=(200, 6)))
    b = DCQFExtractor().fit(rng.uniform(0, np.pi, size=(200, 6)) ** 2)
    assert not np.allclose(a.couplings_, b.couplings_)


def test_scrambled_control_preserves_distribution():
    """The null must differ from the real arm ONLY in which variable each coupling is
    attached to. Same multiset of values, same output width, same circuit."""
    rng = np.random.default_rng(2)
    X = rng.uniform(0, np.pi, size=(200, 6))
    real = DCQFExtractor(orders_encoded=(2,)).fit(X)
    null = ScrambledDCQF(orders_encoded=(2,)).fit(X)
    assert np.allclose(np.sort(real.couplings_), np.sort(null.couplings_))
    assert real.transform(X[:5]).shape == null.transform(X[:5]).shape
    assert not np.allclose(real.transform(X[:5]), null.transform(X[:5]))


def test_chain_product_control_matches_dcqf_width():
    """A dimension-matched control that is not actually dimension-matched is worse
    than no control, so the widths are asserted rather than assumed."""
    rng = np.random.default_rng(4)
    X = rng.uniform(0, np.pi, size=(60, 8))
    q = DCQFExtractor(orders_encoded=(2,), orders_read=(1, 2, 3)).fit(X)
    c = ChainProductControl(orders=(1, 2, 3)).fit(X)
    assert q.n_output_features() == c.n_output_features() == 24
    assert q.n_trainable_params() == c.n_trainable_params() == 0


def test_scramble_is_reproducible():
    rng = np.random.default_rng(6)
    X = rng.uniform(0, np.pi, size=(150, 5))
    a = ScrambledDCQF(seed=42).fit(X).transform(X[:4])
    b = ScrambledDCQF(seed=42).fit(X).transform(X[:4])
    assert np.allclose(a, b)


# ------------------------------------------------------------------- circuits
def test_dense_angle_uses_both_features_per_qubit():
    """The classic silent bug: RZ-encoded features are invisible to a Z-basis readout
    unless something rotates them into the computational basis. If this test starts
    passing trivially (fidelity 1.0), half the input columns are being ignored."""
    a = np.array([0.7, 0.0, 1.2, 0.4])
    b = np.array([0.7, 2.9, 1.2, 0.4])          # differs only in an RZ slot
    assert fidelity(prepare(a, 2), prepare(b, 2)) < 0.99


def test_fidelity_shortcut_equals_compute_uncompute():
    """The simulation shortcut must equal the circuit you would actually run."""
    rng = np.random.default_rng(9)
    for _ in range(25):
        x = rng.uniform(0, np.pi, size=8)
        z = rng.uniform(0, np.pi, size=8)
        assert abs(fidelity(prepare(x, 4), prepare(z, 4))
                   - fidelity_compute_uncompute(x, z, 4)) < 1e-12


def test_gram_matrix_is_a_valid_kernel():
    rng = np.random.default_rng(10)
    K = gram_matrix(rng.uniform(0, np.pi, size=(30, 8)), None, 4)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1.0)
    assert np.linalg.eigvalsh(K).min() > -1e-9, "fidelity kernel must be PSD"


def test_cross_kernel_block_matches_joint_gram():
    """A transposed cross-kernel scores the wrong patients and still produces a
    plausible AUC, so the block is checked against a jointly-computed Gram."""
    rng = np.random.default_rng(12)
    A = rng.uniform(0, np.pi, size=(9, 8))
    B = rng.uniform(0, np.pi, size=(14, 8))
    J = gram_matrix(np.vstack([A, B]), None, 4)
    assert np.allclose(gram_matrix(A, B, 4, symmetric=False), J[:9, 9:], atol=1e-12)


def test_n_weights_agrees_with_ansatz_module():
    """One definition of the parameter count. Two would eventually disagree, and the
    number ends up on a slide."""
    for nq, d in [(4, 6), (3, 2), (5, 4)]:
        assert n_weights(nq, d) == int(np.prod(weight_shape(nq, d)))
    assert n_weights(4, 6) == 96


# ------------------------------------------------------------------ gradients
def test_parameter_shift_matches_finite_difference():
    from qheart.quantum.ansatz import init_weights
    rng = np.random.default_rng(13)
    v = VQC(n_qubits=3, depth=2)
    w = np.asarray(init_weights(3, 2, seed=1), float).reshape(weight_shape(3, 2))
    x = rng.uniform(0, np.pi, size=6)
    g_ps = v._grad_z0(x, w)
    v.finite_diff = True
    g_fd = v._grad_z0(x, w)
    assert np.max(np.abs(g_ps)) > 1e-3, "degenerate point: the test would prove nothing"
    assert np.max(np.abs(g_ps - g_fd)) < 1e-6


# ---------------------------------------------------------------------- heads
def test_logistic_head_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 4))
    w, b = np.array([1.5, -2.0, 0.5, 0.0]), 0.3
    y = (rng.uniform(size=4000) < sigmoid(X @ w + b)).astype(int)
    h = LogisticHead(C=1e6).fit(X, y)
    assert h.converged_
    assert np.max(np.abs(h.coef_ - w)) < 0.2 and abs(h.intercept_ - b) < 0.2


def test_kernel_head_agrees_with_primal_on_linear_kernel():
    """The representer theorem as an executable check."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = (X @ rng.normal(size=5) > 0).astype(int)
    dk = KernelLogisticHead(C=1.0).fit(X @ X.T, y).decision_function(X @ X.T)
    dp = LogisticHead(C=1.0).fit(X, y).decision_function(X)
    assert np.corrcoef(dk, dp)[0, 1] > 0.999


def test_kernel_head_parameter_count_grows_with_n():
    """The headline honesty check for the QSVC row: a kernel method's model size is
    O(N), not the circuit's angle count."""
    rng = np.random.default_rng(2)
    for n in (50, 200):
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] > 0).astype(int)
        assert KernelLogisticHead().fit(X @ X.T, y).n_trainable_params() == n + 1


@pytest.mark.parametrize("bad", ["labels", "one_class", "asymmetric_kernel"])
def test_head_guards_fire(bad):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 3))
    with pytest.raises(ValueError):
        if bad == "labels":
            LogisticHead().fit(X, np.arange(20))
        elif bad == "one_class":
            LogisticHead().fit(X, np.zeros(20, dtype=int))
        else:
            KernelLogisticHead().fit(np.array([[1.0, 2.0], [3.0, 1.0]]),
                                     np.array([0, 1]))


# ----------------------------------------------------------------- algorithms
def test_three_algorithms_share_one_protocol():
    """All three must satisfy the harness ``Model`` protocol identically, because the
    fairness of the results table rests on there being exactly one scoring path."""
    rng = np.random.default_rng(14)
    X = rng.uniform(0, np.pi, size=(80, 8))
    y = (X.sum(1) > np.median(X.sum(1))).astype(int)
    for m in (QSVC(n_qubits=4),
              HybridQNN(n_qubits=4, depth=3),
              VQC(n_qubits=4, depth=1, epochs=1, batch_size=40)):
        m.fit(X[:60], y[:60])
        P = m.predict_proba(X[60:])
        assert P.shape == (20, 2)
        assert np.allclose(P.sum(axis=1), 1.0)
        assert P.min() >= 0.0 and P.max() <= 1.0
        assert m.n_trainable_params() > 0


def test_qsvc_labels_its_own_head_honestly():
    """When sklearn is absent the fallback is NOT an SVC, and the object has to say so
    rather than leaving it to whoever writes the results table."""
    rng = np.random.default_rng(15)
    X = rng.uniform(0, np.pi, size=(40, 8))
    y = (X[:, 0] > np.median(X[:, 0])).astype(int)
    q = QSVC(n_qubits=4).fit(X, y)
    assert ("SVC" in q.head_kind_) or ("NOT an SVC" in q.head_kind_)
    assert q.n_quantum_params() == 0
    assert q.n_trainable_params() >= len(y)


def test_frozen_hybrid_has_zero_quantum_params():
    """ADR-008: the shipped default is a fixed extractor, so the quantum stage
    contributes nothing trainable and the matched control is a zero-parameter map."""
    rng = np.random.default_rng(16)
    X = rng.uniform(0, np.pi, size=(40, 8))
    y = (X[:, 1] > np.median(X[:, 1])).astype(int)
    h = HybridQNN(n_qubits=4, depth=6, trainable=False).fit(X, y)
    assert h.n_quantum_params() == 0
    assert h.n_trainable_params() == h.head_.n_trainable_params()


def test_trainable_hybrid_refuses_rather_than_pretends():
    rng = np.random.default_rng(17)
    X = rng.uniform(0, np.pi, size=(20, 8))
    with pytest.raises(NotImplementedError):
        HybridQNN(trainable=True).fit(X, (X[:, 0] > 1.5).astype(int))


def test_vqc_training_reduces_loss():
    rng = np.random.default_rng(18)
    X = rng.uniform(0, np.pi, size=(60, 8))
    y = (np.cos(X[:, 0]) * np.cos(X[:, 1]) > 0).astype(int)
    v = VQC(n_qubits=4, depth=2, epochs=4, batch_size=20, lr=0.2).fit(X, y)
    assert v.loss_[-1] < v.loss_[0]


def test_vqc_refuses_to_drop_features():
    """Silently dropping columns that do not fit the encoding is how a model ends up
    'ignoring' half the clinical variables without anyone noticing."""
    rng = np.random.default_rng(19)
    X = rng.uniform(0, np.pi, size=(20, 12))       # 12 > 2*4
    with pytest.raises(ValueError, match="exceed"):
        VQC(n_qubits=4, epochs=1).fit(X, (X[:, 0] > 1.5).astype(int))
