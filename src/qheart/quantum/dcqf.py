"""Digitized counterdiabatic quantum feature extraction (DCQF).

Implements Simen, Flores-Garrigós, De Oliveira, Alvarado Barrios, Gomez Cadavid,
Dalal, Solano, Hegade & Zhang, *Digitized counterdiabatic quantum feature
extraction*, Sci Rep (2026), doi:10.1038/s41598-026-67564-0 (Kipu Quantum).
**Accepted-article-in-press at the time of writing** -- the authors state the text may
still change, so cite it with that status and re-check the equations against the final
version before submission.

The method in four steps, Eq. (1)-(4) of the paper:

1. Encode a feature vector into the **longitudinal fields** of a spin-glass
   Hamiltonian, and the *statistical correlations between features* into its k-body
   **couplings**:  H(x) = sum_i x_i Z_i + sum_k sum_{S in G(k)} c_S prod_{i in S} Z_i
2. Build an adiabatic sweep from a transverse field to that Hamiltonian, and add the
   first-order **counterdiabatic** (adiabatic gauge potential) term.
3. Evolve for a **single Trotter step in the impulse regime** (Nsteps=1, dt=1).
4. Read out **one-, two- and three-body Pauli-Z string expectations** as the new
   features.

Why this is a good fit for a heart-disease project, stated honestly:

* It is a **feature extractor, not a classifier**. Its output is a feature matrix for
  an ordinary classical model. So it composes with the classifiers this repo already
  has instead of competing with them, and it is scored through the same harness.
* It has **no trainable quantum parameters at all**. The couplings come from mutual
  information measured on the training fold; the AGP coefficient is analytic. So the
  usual "how many parameters does the quantum part have" question has the cleanest
  possible answer: zero. That makes the parameter-matched control easy to construct
  and impossible to fudge.
* One circuit per sample, all observables commuting, so a full dataset is O(N)
  circuits -- unlike the O(N^2) of a fidelity kernel. For 918 patients that is 918
  circuits rather than ~421k evaluations.

**The honest caveats, which belong on a slide and not in a footnote:**

The paper's own numbers are modest and its best results come from a SHAP-selected
*hybrid* set that includes classical features -- quantum features alone lose to the
classical baseline on the breast dataset (F1 0.754 vs 0.778). Their gain is also
measured on 171-sample and 702-sample datasets where a standard deviation of +-0.09
swamps most differences. And DCQF *expands* dimensionality: 8 features in can become
dozens out, so comparing it against 8 raw features is not a fair test. That is what
``qheart.models.dcqf_control`` exists to correct.

Four deviations from the paper are implemented deliberately and listed in full by
``DCQFExtractor.provenance()``: a surrogate AGP coefficient, the identity
variable-to-qubit assignment instead of their connectivity-aware genetic algorithm
(Supplementary Sec. 1 -- simulation has no coupling map, and ``qubit_assignment=``
accepts a permutation if you later run on hardware), exact simulation rather than IBM
Kingston, and a midpoint evaluation of the schedule derivative. Print that string next
to any DCQF number you report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

import numpy as np

from qheart.quantum.statevector import State, expectations_from_counts, guard_qubits

__all__ = ["mutual_information_matrix", "coupling_coefficients", "agp_alpha",
           "schedule_derivative", "hypergraph_closed_chain", "z_string_observables",
           "DCQFExtractor", "DCQF_READOUT_ORDERS"]

DCQF_READOUT_ORDERS = (1, 2, 3)


# --------------------------------------------------------------------- step 1
def _discretise(col: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin a column for discrete mutual information.

    Quantiles rather than equal width: clinical variables are skewed (cholesterol,
    oldpeak), and equal-width bins put almost every patient in one bin, which drives
    the MI estimate to zero and silently switches the couplings off.

    Ties are common in clinical data (many patients share oldpeak == 0.0), so
    ``np.unique`` on the edges is required -- duplicate edges would create empty bins
    and bias the estimate.
    """
    col = np.asarray(col, dtype=float)
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return np.zeros(col.shape, dtype=int)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(finite, qs))
    return np.digitize(col, edges).astype(int)


def mutual_information_matrix(X: np.ndarray, n_bins: int = 4,
                              normalise: bool = True) -> np.ndarray:
    """Pairwise empirical mutual information I(x_a, x_b), in nats.

    MI is computed on the plug-in (maximum-likelihood) estimator over a discretised
    joint histogram. That estimator is **biased upward** on small samples -- with
    n_bins=4 there are 16 joint cells, and on a 700-row training fold some cells hold
    a handful of patients, so even independent variables show a small positive MI.
    The bias is identical across all pairs and the values are rescaled below, so it
    does not distort the *relative* coupling structure, which is what the encoding
    uses. Do not quote these numbers as information-theoretic measurements.

    With ``normalise=True`` the matrix is divided by its own maximum, so couplings
    land in [0, 1] and the Hamiltonian's energy scale stays comparable to the field
    terms (which are angles in [0, pi]). Without it, a dataset with strongly
    dependent features produces couplings that dominate the fields entirely, and the
    extracted features stop depending on the individual patient.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
    n, d = X.shape
    if n < 2:
        raise ValueError("mutual information needs at least 2 samples")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    D = np.stack([_discretise(X[:, j], n_bins) for j in range(d)], axis=1)
    n_states = int(D.max()) + 1
    M = np.zeros((d, d), dtype=float)

    for a in range(d):
        for b in range(a + 1, d):
            joint = np.zeros((n_states, n_states), dtype=float)
            np.add.at(joint, (D[:, a], D[:, b]), 1.0)
            joint /= joint.sum()
            pa = joint.sum(axis=1, keepdims=True)
            pb = joint.sum(axis=0, keepdims=True)
            denom = pa @ pb
            nz = (joint > 0) & (denom > 0)
            mi = float(np.sum(joint[nz] * np.log(joint[nz] / denom[nz])))
            M[a, b] = M[b, a] = max(mi, 0.0)      # clip estimator noise below zero

    if normalise:
        mx = M.max()
        if mx > 0:
            M = M / mx
    return M


def coupling_coefficients(mi: np.ndarray, subsets: Sequence[Sequence[int]]) -> np.ndarray:
    """Eq. (2): c_S = binom(|S|,2)^-1 * sum_{{a,b} subset S} I(x_a, x_b).

    The mean pairwise MI within the subset. For |S| == 2 that is just I(x_a, x_b);
    for |S| == 3 it averages the three pairs. Note this makes a three-body coupling
    an *average of pairwise* dependencies -- it does not measure genuine three-way
    interaction information. That is what the paper specifies, and it is worth knowing
    when someone asks whether the three-body terms carry irreducibly higher-order
    information: by construction, they do not.
    """
    mi = np.asarray(mi, dtype=float)
    out = np.empty(len(subsets), dtype=float)
    for i, S in enumerate(subsets):
        S = list(S)
        if len(S) < 2:
            raise ValueError(f"subset {S} must have at least 2 members")
        pairs = list(combinations(S, 2))
        out[i] = float(np.mean([mi[a, b] for a, b in pairs]))
    return out


def hypergraph_closed_chain(n: int, k: int) -> list[tuple[int, ...]]:
    """G(k): k-body subsets along a closed linear chain, i.e. (i, i+1, ..., i+k-1) mod n.

    The paper evaluates higher-body correlations "along a closed linear chain on the
    device, which preserves the original feature dimensionality" -- n subsets for n
    variables, so the feature count per order stays linear instead of combinatorial.
    The alternative, all C(n,3) triples, is 56 terms for 8 features and grows as n^3.
    """
    if n < 2:
        raise ValueError("need at least 2 variables")
    if k < 2:
        raise ValueError("k must be >= 2")
    if k > n:
        return []
    if n == k:
        return [tuple(range(n))]
    return [tuple((i + j) % n for j in range(k)) for i in range(n)]


def z_string_observables(n: int, orders: Sequence[int] = DCQF_READOUT_ORDERS
                         ) -> list[tuple[int, ...]]:
    """The Z strings measured as output features: singles, then chain pairs, then triples.

    Order is fixed and deterministic so that column j of the output always means the
    same observable across folds and runs. A set-based implementation here would
    reorder columns between Python versions and make SHAP importances meaningless.
    """
    obs: list[tuple[int, ...]] = []
    for k in sorted(set(int(o) for o in orders)):
        if k == 1:
            obs.extend((i,) for i in range(n))
        else:
            obs.extend(hypergraph_closed_chain(n, k))
    return obs


# --------------------------------------------------------------------- step 2
def schedule_derivative(t: float, total_time: float) -> float:
    """lambda'(t) for the sweep schedule lambda(t) = sin^2(pi t / 2T).

    The counterdiabatic term is proportional to the *rate* at which the Hamiltonian
    changes, so this factor sets the entire size of the AGP kick. It is worth its own
    function because of a trap:

        lambda'(t) = (pi / 2T) sin(pi t / T)   ->   lambda'(T) = 0

    The schedule is flat at both endpoints (that is the point of a smooth ramp), so
    evaluating the derivative at the *end* of the single Trotter step -- which is what
    ``Nsteps=1, dt=1, T=1`` literally suggests -- multiplies the counterdiabatic term
    by exactly zero and silently deletes it. The circuit then reduces to diagonal Z
    phases on |+>^n, which leaves the probabilities uniform and every measured
    expectation exactly 0.0. That failure is completely silent: the pipeline runs, the
    features are all zeros, and the classifier reports chance accuracy.

    So the step is evaluated at its **midpoint**, t = dt/2, the standard convention for
    a first-order product formula. This is a declared deviation; see
    ``DCQFExtractor.provenance()`` item 4.
    """
    if total_time <= 0:
        raise ValueError("total_time must be positive")
    return float((np.pi / (2.0 * total_time)) * np.sin(np.pi * t / total_time))


def agp_alpha(fields: np.ndarray, couplings: np.ndarray) -> float:
    """Analytic first-order adiabatic-gauge-potential coefficient.

    The paper cites [39] for "alpha can be computed analytically for arbitrary
    spin-glass problems" and does not reproduce the formula. The standard first-order
    variational result minimises the action ||dH - i[A, H]||^2 and gives, for a
    transverse-field-to-spin-glass sweep, a coefficient whose denominator is the
    squared energy scale of the encoded Hamiltonian, with lattice-dependent prefactors.

    **Rather than assert a formula this project cannot verify, this returns a bounded,
    smooth surrogate** with the properties the protocol needs: alpha shrinks as the
    encoded energy scale grows, and it never blows up when the Hamiltonian is near
    zero. This is a declared deviation, flagged in ``DCQFExtractor.provenance()``.

    **The denominator uses the MEAN squared coefficient, not the sum.** This is not
    cosmetic and the first version of this function got it wrong. With a sum, ||h||^2
    grows linearly in the number of features, so alpha decays as 1/n and the
    counterdiabatic rotation angle collapses: measured on 8 features in [0, pi] the
    kick was 4.3 degrees, and at 22 features 1.5 degrees. At angles that small
    sin(theta) ~ theta, the whole map linearises and the extracted features become a
    faint linear echo of the inputs -- the quantum step stops contributing anything a
    matrix multiply could not. A variational *single-site* AGP ansatz is set by the
    LOCAL field on that site, which is an intensive quantity, so the mean is also the
    better-motivated choice physically. It keeps the kick scale-free in n: roughly
    32 degrees for the same data, whether there are 4 features or 22.

    Sanity-check the output of this function before trusting a DCQF result. If alpha
    is of order 1e-3, the extractor is a no-op with extra steps.
    """
    h = np.asarray(fields, dtype=float).ravel()
    j = np.asarray(couplings, dtype=float).ravel()
    h2 = float(np.mean(np.square(h))) if h.size else 0.0
    j2 = float(np.mean(np.square(j))) if j.size else 0.0
    return -0.5 / (1.0 + 2.0 * h2 + 2.0 * j2)


# --------------------------------------------------------------------- steps 3-4
@dataclass
class DCQFExtractor:
    """Hamiltonian encoding -> counterdiabatic evolution -> Z-string features.

    Scikit-learn transformer interface, so it drops into a ``Pipeline`` as a *step*
    and therefore cannot see test rows during ``fit`` -- the repo's rule 1. This
    matters more here than for most preprocessing: the couplings are estimated from
    data, so a globally-fitted DCQF would leak the test fold's correlation structure
    into every training fold. That leak would *raise* accuracy and be invisible in
    the metrics.

    Parameters
    ----------
    orders_encoded : which k-body terms appear in H(x). (2,) reproduces the paper's
        first dynamics; (3,) the second; (2, 3) puts both in one Hamiltonian.
    orders_read : which Z-string orders are measured as features.
    n_bins : quantile bins for the MI estimate.
    shots : ``None`` for exact expectation values (infinite-shot limit), or an
        integer to simulate hardware shot noise. The paper used 8000.
    trotter_steps : kept at 1, the impulse regime the paper operates in.
    agp_scale : multiplier on the counterdiabatic kick. 1.0 is the surrogate alpha
        unmodified and is the pre-registered default. This knob exists because the
        kick angle is what decides whether the extractor is a genuine nonlinear map or
        a linear echo, and that deserves to be visible and sweepable rather than
        buried in ``agp_alpha``. **It is a hyperparameter: tune it inside the training
        fold or fix it in advance, never against the test set.** ``diagnostics()``
        reports the resulting angle so you can check it before running anything.
    include_input : concatenate the original features onto the quantum ones. The
        paper's best results ("Hybrid") do exactly this before SHAP selection.
    """

    orders_encoded: tuple[int, ...] = (2,)
    orders_read: tuple[int, ...] = DCQF_READOUT_ORDERS
    n_bins: int = 4
    shots: int | None = None
    trotter_steps: int = 1
    dt: float = 1.0
    agp_scale: float = 1.0
    include_input: bool = False
    qubit_assignment: Sequence[int] | None = None
    seed: int = 20260830
    scramble_couplings: bool = False        # control knob; see models/dcqf_control.py

    mi_: np.ndarray | None = field(default=None, init=False, repr=False)
    subsets_: list[tuple[int, ...]] = field(default_factory=list, init=False, repr=False)
    couplings_: np.ndarray | None = field(default=None, init=False, repr=False)
    observables_: list[tuple[int, ...]] = field(default_factory=list, init=False, repr=False)
    n_features_in_: int | None = field(default=None, init=False)

    # -------------------------------------------------------------- sklearn API
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "DCQFExtractor":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"expected 2-D input, got shape {X.shape}")
        n_feat = X.shape[1]
        guard_qubits(n_feat)                 # one qubit per feature
        if self.trotter_steps != 1:
            raise NotImplementedError(
                "only the single-step impulse regime (trotter_steps=1) is implemented, "
                "matching the paper. Multi-step needs the full A(t)/B(t) schedule "
                "evaluated at each step, which is a different code path -- add it "
                "deliberately rather than by loosening this check."
            )
        self.n_features_in_ = n_feat
        self.mi_ = mutual_information_matrix(X, n_bins=self.n_bins)

        self.subsets_ = []
        for k in sorted(set(int(o) for o in self.orders_encoded)):
            self.subsets_.extend(hypergraph_closed_chain(n_feat, k))
        self.couplings_ = coupling_coefficients(self.mi_, self.subsets_) \
            if self.subsets_ else np.zeros(0)

        if self.scramble_couplings:
            # Destroy the data-derived structure while preserving the value
            # distribution exactly. This is the null model: same circuit, same
            # coupling magnitudes, correlations no longer matched to the variables.
            rng = np.random.default_rng(self.seed + 991)
            self.couplings_ = rng.permutation(self.couplings_)

        self.observables_ = z_string_observables(n_feat, self.orders_read)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.n_features_in_ is None:
            raise RuntimeError("call fit before transform")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"transform got {X.shape[1] if X.ndim == 2 else '?'} features but fit "
                f"saw {self.n_features_in_}. The couplings are indexed by feature "
                f"position, so a column reorder between fit and transform would "
                f"silently pair the wrong variables."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("DCQF received NaN or inf; impute inside the fold first")

        out = np.empty((X.shape[0], len(self.observables_)), dtype=float)
        for i, row in enumerate(X):
            out[i] = self._evolve_and_measure(row)
        return np.hstack([X, out]) if self.include_input else out

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # -------------------------------------------------------------- the physics
    def _wire(self, i: int) -> int:
        return i if self.qubit_assignment is None else int(self.qubit_assignment[i])

    def _evolve_and_measure(self, x: np.ndarray) -> np.ndarray:
        """One sample -> one circuit -> all Z-string expectations.

        The circuit, in the impulse regime with Nsteps=1:

          |+>^n                              ground state of H_i = -sum_j X_j
          exp(-i dt H(x))                    the encoding Hamiltonian, all Z terms
          exp(-i dt A)                       first-order counterdiabatic correction

        **The counterdiabatic term is not optional here, it is the only thing that
        does anything.** H(x) is diagonal, and |+>^n is the uniform superposition, so
        diagonal gates change only phases and leave every measurement probability at
        2^-n. Every Z-string expectation of ``h_all() -> diagonal gates`` is therefore
        exactly zero, for every patient. All of the encoded information reaches the
        readout through A, which contains the off-diagonal Y terms. Any bug that
        zeroes alpha or lambda' produces an all-zeros feature matrix rather than a
        degraded one -- see ``tests`` for the guard against exactly that.

        The AGP, derived rather than assumed:

          H_ad(t) = A(t) H_i + B(t) H_z,  A = 1 - lambda,  B = lambda
          dH_ad/dt = lambda' (H_z - H_i)
          [H_ad, dH_ad/dt] = lambda' (A + B) [H_i, H_z] = lambda' [H_i, H_z]

        and with H_i = -sum_j X_j, using [X, Z] = -2i Y,

          [H_i, H_z] = 2i ( sum_j x_j Y_j
                            + sum_S sum_{j in S} c_S Y_j prod_{i in S, i != j} Z_i )

        so A = i alpha [H_ad, dH_ad/dt] = -2 alpha lambda' (the same bracket).

        Each term is a Pauli string P with P^2 = I, so exp(-i dt g P) = exp(-i th/2 P)
        with th = 2 dt g, giving th = -4 alpha lambda' dt * (x_j or c_S).

        Note what the k-body part is: **one Y on the kicked site and Z on the rest of
        the subset**, not a single-qubit rotation whose angle was summed over the
        couplings. The scalar-weight version this replaced looked reasonable and was
        wrong -- it threw away the entanglement the coupling terms are supposed to
        create, and produced features weaker than the raw inputs.

        Two approximations remain, both inherent to "digitized" and both declared:

        * The AGP terms do not all commute with each other (Y_0 and Y_1 Z_0 anticommute),
          so applying them one after another is a first-order product formula.
        * A does not commute with H(x), so splitting the two exponentials is likewise
          first order. Per the paper this residual non-adiabatic mixing is the feature
          map, not an error to be driven to zero.
        """
        n = self.n_features_in_
        st = State(n).h_all()

        # --- exp(-i dt H(x)): fields then k-body couplings.
        # rzz_string applies exp(-i theta/2 * Zs), so theta = 2 * dt * coefficient.
        for i, xi in enumerate(x):
            if xi != 0.0:
                st.rzz_string([self._wire(i)], 2.0 * self.dt * float(xi))
        for S, c in zip(self.subsets_, self.couplings_):
            if c != 0.0:
                st.rzz_string([self._wire(j) for j in S], 2.0 * self.dt * float(c))

        # --- exp(-i dt A), first-order AGP.
        alpha = agp_alpha(x, self.couplings_)
        total_time = self.trotter_steps * self.dt
        lam_dot = schedule_derivative(self.dt / 2.0, total_time)     # midpoint, not t=T
        scale = -4.0 * alpha * lam_dot * self.dt * self.agp_scale    # theta per unit coefficient

        for j, xj in enumerate(x):                                   # sum_j x_j Y_j
            if xj != 0.0:
                st.ryz_string(self._wire(j), (), scale * float(xj))
        for S, c in zip(self.subsets_, self.couplings_):             # sum_S sum_{j in S}
            if c == 0.0:
                continue
            theta = scale * float(c)
            for j in S:
                st.ryz_string(self._wire(j),
                              [self._wire(i) for i in S if i != j], theta)

        if self.shots is None:
            return st.z_string_expectations(self.observables_)
        counts = st.sample_counts(self.shots, seed=self.seed)
        return expectations_from_counts(counts, n, self.observables_)

    # -------------------------------------------------------------- accounting
    def diagnostics(self, X: np.ndarray) -> dict[str, float]:
        """Is this extractor actually doing anything? Call it before every run.

        DCQF has two silent failure modes, both of which produce a pipeline that runs
        cleanly and reports plausible-looking chance-level results:

        1. **Dead**: alpha or lambda' is zero, so the circuit is diagonal, the
           probabilities stay uniform at 2^-n, and every feature is exactly 0.0.
        2. **Linear**: the kick angle is a couple of degrees, so sin(theta) ~ theta
           and the extractor is an expensive linear map. Nothing looks wrong; the
           quantum step simply is not contributing.

        Neither raises. So this returns the numbers that distinguish them --
        ``agp_angle_deg`` and ``feature_std`` are the two to look at. A healthy
        configuration on angle-scaled clinical data is tens of degrees and a feature
        standard deviation comfortably above the shot noise you plan to run at.
        """
        if self.n_features_in_ is None:
            raise RuntimeError("call fit first")
        X = np.asarray(X, dtype=float)
        alphas = np.array([agp_alpha(row, self.couplings_) for row in X])
        lam_dot = schedule_derivative(self.dt / 2.0, self.trotter_steps * self.dt)
        scale = -4.0 * alphas * lam_dot * self.dt * self.agp_scale
        typ_coef = float(np.mean(np.abs(X))) if X.size else 0.0
        Z = self.transform(X)
        if self.include_input:
            Z = Z[:, self.n_features_in_:]
        return {
            "alpha_mean": float(np.mean(alphas)),
            "lambda_dot": float(lam_dot),
            "agp_angle_deg": float(np.degrees(np.mean(np.abs(scale)) * typ_coef)),
            "feature_std": float(np.mean(np.std(Z, axis=0))),
            "feature_absmax": float(np.max(np.abs(Z))) if Z.size else 0.0,
            "shot_noise_at_8000": 1.0 / np.sqrt(8000.0),
        }

    def n_output_features(self) -> int:
        if self.n_features_in_ is None:
            raise RuntimeError("call fit first")
        return len(self.observables_) + (self.n_features_in_ if self.include_input else 0)

    def feature_names(self) -> list[str]:
        names = [("Z" + "Z".join(str(i) for i in S)) for S in self.observables_]
        if self.include_input:
            names = [f"x{i}" for i in range(self.n_features_in_)] + names
        return names

    def n_trainable_params(self) -> int:
        """Zero. Not "small" -- zero.

        The couplings are *measured* from the training fold, like a mean and variance
        in a StandardScaler, and the AGP coefficient is computed analytically. Nothing
        here is optimised against a loss. That is the cleanest version of this
        project's central claim, and it is why the matched control for this arm is a
        fixed transform rather than a trained one.
        """
        return 0

    def circuit_cost(self) -> dict[str, int]:
        """Gate and circuit counts, for the cost slide. One circuit per sample.

        Counted on the *hardware* decomposition, not on this simulator, because the
        simulator's diagonal shortcut would flatter the numbers: here a 3-body Z phase
        is one array multiply, on a device it is a four-CNOT ladder.

        The AGP is the expensive half and it is easy to undercount. Each k-body subset
        contributes k separate ``Y_j (x) Z^(k-1)`` terms -- one per member -- so a
        3-body coupling costs three rotations, not one.
        """
        if self.n_features_in_ is None:
            raise RuntimeError("call fit first")
        n = self.n_features_in_
        # Encoding: each k-body Z string is a 2*(k-1) CNOT ladder around one RZ.
        enc_cnots = sum(2 * (len(S) - 1) for S in self.subsets_)
        enc_rot = n + len(self.subsets_)
        # AGP: n single-qubit Y kicks, plus k terms per k-body subset, each a
        # Y (x) Z^(k-1) string = same 2*(k-1) ladder plus two basis-change gates.
        agp_terms = sum(len(S) for S in self.subsets_)
        agp_cnots = sum(len(S) * 2 * (len(S) - 1) for S in self.subsets_)
        agp_rot = n + agp_terms
        return {"qubits": n,
                "circuits_per_sample": 1,
                "z_string_terms": enc_rot,
                "agp_terms": n + agp_terms,
                "cnots_hardware": enc_cnots + agp_cnots,
                "single_qubit_rotations": enc_rot + agp_rot + 2 * agp_terms + n,
                "observables_measured": len(self.observables_),
                "output_features": self.n_output_features()}

    def provenance(self) -> str:
        """What is reproduced, what is not. Print this next to any DCQF result."""
        return (
            "DCQF after Simen et al., Sci Rep (2026) doi:10.1038/s41598-026-67564-0 "
            "[accepted article in press].\n"
            f"  encoded orders {tuple(self.orders_encoded)}, read orders "
            f"{tuple(self.orders_read)}, MI bins {self.n_bins}, "
            f"shots {self.shots if self.shots else 'exact (infinite-shot limit)'}\n"
            "  DEVIATIONS from the published protocol, declared:\n"
            "   1. AGP coefficient alpha uses a bounded scale-invariant surrogate, not "
            "the analytic formula of their ref. [39] (see agp_alpha docstring).\n"
            "   2. Variable-to-qubit assignment is the identity, not their "
            "connectivity-aware genetic algorithm -- simulation has no coupling map.\n"
            "   3. Exact statevector simulation, not IBM Kingston hardware. No noise, "
            "no error mitigation, no transpilation.\n"
            "   4. The schedule derivative lambda' is evaluated at the step MIDPOINT. "
            "The paper's Nsteps=1, dt=1 read literally puts it at t=T, where lambda' "
            "is identically zero and the counterdiabatic term vanishes.\n"
            "  Do not describe results from this module as reproducing the paper's "
            "hardware numbers."
        )
