"""Does the DCQF quantum step do anything its own null model does not?

This is the experiment the Kipu paper does not run, and it is the one that decides
whether the DCQF arm belongs in the SIH deck as a result or as a negative finding.

The question is *not* "does DCQF beat raw features". Of course it can: it turns 8
columns into 24, and more columns fit more label. The question is whether the two
things DCQF specifically claims -- the mutual-information coupling encoding of Eq. (2)
and the counterdiabatic quantum evolution -- contribute anything beyond a generic
nonlinear expansion of the same width. So each DCQF arm is run against two matched
comparators on byte-identical folds:

  scrambled DCQF   the same circuit with the couplings permuted off their variables.
                   Same evolution, same coupling magnitudes, same output width; only
                   the variable-to-coupling correspondence is destroyed. Isolates
                   Eq. (2).
  chain products   classical products over the same closed-chain subsets the circuit
                   measures, tanh-squashed to the same [-1, 1] range. Same width, no
                   circuit at all. Isolates the quantum step.

Four label regimes, because the answer depends on the data and saying "DCQF works"
without saying "on what" is how the literature got here:

  linear       y from a linear score. No interaction structure at all.
  pairwise     y from a sum of cos(x_i)cos(x_i+1) products. Built to suit DCQF: pure
               pairwise structure, no linear signal. This is DCQF's best case.
  mixed        mostly linear with a pairwise kicker.
  cardio-like  features with the *structure* of the eight cardio_train columns that
               enter the circuit -- a correlated systolic/diastolic pair, a correlated
               height/weight pair, skewed ordinal cholesterol and glucose, a rare
               binary smoking flag -- with the label noise calibrated so a logistic
               regression lands on the ~0.79 AUC ceiling measured on the real file.
               **This is not cardio_train.** It is a structural stand-in, and the only
               claim made for it is that iid uniforms are a worse model of clinical
               data than this is. Running the real file remains outstanding.

-------------------------------------------------------------------------------
On proving a negative
-------------------------------------------------------------------------------

An earlier version of this script reported "not significant" at four seeds and three
degrees of freedom, and that is a much weaker statement than it looks: a large
p-value on four observations is a statement about the sample size, not about DCQF.
Three things fix it, and all three are here.

**1. Twenty independent dataset draws instead of four.** Each seed is a fresh X and a
fresh y, so the per-seed observations are genuinely independent and an ordinary
paired t applies -- no Nadeau-Bengio correction, which exists for training-set
overlap between folds and would only throw away the power the extra draws bought.
Nineteen degrees of freedom instead of three moves the critical value from 3.182 to
2.093.

**2. An equivalence test, not just a significance test.** ``tost_equivalence``
answers "how large an effect can we exclude" rather than "did we find one". The
number it reports, ``bound``, is the smallest margin at which the data support an
equivalence claim, and it is what should be quoted: "any effect above X AUC is
excluded at 95%" is a real result, whereas "p > 0.05" is not.

**3. A spiked positive control.** The load-bearing objection to any null is that the
measurement was too blunt to see anything. So one arm is DCQF with a known fraction
of the true latent score mixed into a single column, swept over several magnitudes.
That arm is *not a model* -- it cheats, deliberately, by construction -- it exists
only to answer "if there were a real effect of size s, would this harness find it?"
The smallest injected lift that comes back significant is the harness's empirical
minimum detectable effect, measured through the identical pipeline rather than
assumed from a power formula. Quote it next to the DCQF result: an observed
difference of 0.001 means something very different when the harness is demonstrably
sensitive to 0.01.

Also on the statistics: the **per-fold** t treats 20 folds within a seed as
independent observations. They are not -- folds share ~80% of their training rows --
so that statistic is anti-conservative and is printed only to show direction and
consistency. Never quote it.

Run:  PYTHONPATH=src python scripts/dcqf_null_test.py [--seeds 20] [--n 400]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from qheart.eval.stats import paired_t_test, tost_equivalence        # noqa: E402
from qheart.models.dcqf_control import ChainProductControl           # noqa: E402
from qheart.models.heads import LogisticHead                         # noqa: E402
from qheart.quantum.dcqf import DCQFExtractor                        # noqa: E402

D = 8
ARMS = ["raw X (8)", "DCQF (24)", "DCQF scrambled (24)", "chain products (24)",
        "raw+DCQF (32)", "raw+scrambled (32)", "raw+products (32)"]
REFERENCE = "DCQF (24)"

# Mixing weights for the spiked positive control. 0.0 must reproduce plain DCQF
# exactly, which is the sanity check that the spike machinery is wired up at all.
SPIKES = (0.0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.22)

# Which feature generator each label regime draws from. Regimes sharing a generator
# share their fitted transforms, because DCQF never sees y.
REGIMES = {"linear": "uniform", "pairwise": "uniform", "mixed": "uniform",
           "cardio-like": "cardio"}

# The margin for the headline equivalence claim, fixed in advance. Justification:
# on the real cardio_train, everything machine learning buys over a single
# untrained blood-pressure threshold is about one accuracy point, so an AUC effect
# below 0.01 could not change a clinical decision or a deck claim. The reported
# ``bound`` does not depend on this choice.
MARGIN = 0.01


# ------------------------------------------------------------------- scoring
def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUC. Ties get average ranks, which matters here because several
    arms produce near-duplicate scores."""
    y, s = np.asarray(y), np.asarray(s)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def score(Ztr, ytr, Zte, yte) -> float:
    """One shared head for every arm. Standardisation uses TRAINING statistics only --
    the arms differ in feature scale, and an L2 penalty is scale-sensitive, so without
    this the comparison would partly measure feature magnitude."""
    mu, sd = Ztr.mean(0), Ztr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    h = LogisticHead(C=1.0).fit((Ztr - mu) / sd, ytr)
    return auc(yte, h.predict_proba((Zte - mu) / sd)[:, 1])


def to_angles(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Min-max the features into [0, pi] using training-fold ranges only.

    Required for the cardio-like regime, where the raw columns are in centimetres,
    kilograms and millimetres of mercury: feeding 165 into an RZ phase wraps it
    around the circle several times and destroys the ordering the encoding is
    supposed to carry. Applied to every generator rather than only the one that
    needs it, so the pipeline is identical across regimes.

    Test rows are clipped rather than extrapolated. An out-of-range test value would
    otherwise land outside [0, pi] and alias onto a different angle.
    """
    lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    f = lambda A: np.clip((A - lo) / span, 0.0, 1.0) * np.pi     # noqa: E731
    return f(Xtr), f(Xte)


# -------------------------------------------------------- feature generators
def make_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0, np.pi, size=(n, D))


def make_cardio_like(n: int, rng: np.random.Generator) -> np.ndarray:
    """Eight features shaped like the eight cardio_train columns that get encoded.

    Not a simulation of the dataset and not fitted to it -- a structural stand-in
    built from its documented marginals and dependencies, so that the extractor
    meets correlated pairs, skewed ordinals and a rare binary instead of eight
    independent uniforms. Those three properties are what an MI-based coupling
    encoding is supposed to exploit, and iid uniforms give it nothing to find,
    which would stack the deck toward a null.

    Column order: age, systolic, diastolic, height, weight, cholesterol, glucose,
    smoking.
    """
    z = rng.normal(size=n)                                   # latent risk driver
    age = 53.3 + 6.8 * (0.55 * z + 0.835 * rng.normal(size=n))
    ap_hi = 126.0 + 16.0 * (0.70 * z + 0.714 * rng.normal(size=n))
    ap_lo = 40.0 + 0.35 * ap_hi + 4.0 * rng.normal(size=n)   # tracks systolic, r ~ 0.8
    height = 164.4 + 8.2 * rng.normal(size=n)
    weight = -55.0 + 0.73 * height + 9.5 * (0.25 * z + rng.normal(size=n))

    def ordinal(latent, cuts):
        q = np.quantile(latent, cuts)
        return 1.0 + sum((latent > c).astype(float) for c in q)

    chol = ordinal(0.45 * z + 0.893 * rng.normal(size=n), (0.750, 0.886))
    gluc = ordinal(0.30 * z + 0.954 * rng.normal(size=n), (0.850, 0.925))
    smoke = (rng.uniform(size=n) < 0.088).astype(float)
    return np.column_stack([age, ap_hi, ap_lo, height, weight, chol, gluc, smoke])


# ------------------------------------------------------------------- labels
def latent_score(kind: str, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The noiseless signal. Returned separately from the label because the spiked
    positive control needs it, and because the noise level is what sets the ceiling."""
    n, d = X.shape
    if kind == "linear":
        return X @ rng.normal(size=d)
    if kind == "pairwise":
        return sum(np.cos(X[:, i]) * np.cos(X[:, (i + 1) % d]) for i in range(d))
    if kind == "mixed":
        lin = X @ rng.normal(size=d)
        pair = np.mean([np.cos(X[:, i]) * np.cos(X[:, (i + 1) % d]) for i in range(d)],
                       axis=0)
        return lin + 1.5 * np.std(lin) * pair
    if kind == "cardio-like":
        # Risk rises with systolic pressure, age, cholesterol and BMI -- the ordering
        # the real dataset shows, and near-linear in blood pressure as it is there.
        age, ap_hi, ap_lo, height, weight, chol, gluc, _ = X.T
        bmi = weight / np.square(height / 100.0)
        return (0.055 * (ap_hi - 126.0) + 0.020 * (ap_lo - 84.0)
                + 0.048 * (age - 53.3) + 0.42 * (chol - 1.0)
                + 0.12 * (gluc - 1.0) + 0.035 * (bmi - 27.0))
    raise ValueError(f"unknown label regime {kind!r}")


# Noise multiple per regime. 0.35 is the original setting; the cardio-like value is
# calibrated so a logistic regression on the eight features reaches roughly the
# 0.7937 AUC that logistic regression actually achieves on cardio_train, rather
# than an arbitrarily easy or impossible problem. See --calibrate.
NOISE = {"linear": 0.35, "pairwise": 0.35, "mixed": 0.35, "cardio-like": 1.10}


def make_labels(kind: str, X: np.ndarray, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray]:
    s = latent_score(kind, X, rng)
    noisy = s + NOISE[kind] * np.std(s) * rng.normal(size=X.shape[0])
    return (noisy > np.median(noisy)).astype(int), s


# --------------------------------------------------- spiked positive control
def spike(Ztr, Zte, s_tr, s_te, eps: float, col: int = 0):
    """DCQF features with ``eps`` worth of the true latent signal mixed into one column.

    **This arm is not a model and must never be reported as one.** It has access to
    the noiseless score that generated the labels, which no honest feature extractor
    does. It exists to calibrate the measuring instrument: by sweeping ``eps`` we
    find the smallest genuine AUC improvement this harness can distinguish from
    nothing, which is the number that tells you whether the DCQF null means "no
    effect" or "no resolution".

    The mix is orthogonal-style, ``sqrt(1-eps^2) * column + eps * signal``, so the
    column width is unchanged and at eps=0 the arm is bit-identical to plain DCQF
    after the head's standardisation. Both standardisations use training rows only;
    the latent score is a function of X alone, so no label information crosses the
    fold boundary -- the injected signal is legitimate information about the test
    *features*, which is exactly what a better extractor would have found.
    """
    if eps <= 0.0:
        return Ztr, Zte
    a = float(np.sqrt(max(0.0, 1.0 - eps * eps)))
    mz, sz = Ztr[:, col].mean(), Ztr[:, col].std()
    ms, ss = s_tr.mean(), s_tr.std()
    sz = sz if sz > 1e-12 else 1.0
    ss = ss if ss > 1e-12 else 1.0
    A, B = Ztr.copy(), Zte.copy()
    A[:, col] = a * (Ztr[:, col] - mz) / sz + eps * (s_tr - ms) / ss
    B[:, col] = a * (Zte[:, col] - mz) / sz + eps * (s_te - ms) / ss
    return A, B


# ---------------------------------------------------------------- the run
def collect(n: int, n_seeds: int, folds: int) -> dict:
    """Every arm, every regime, on shared folds.

    The loop is organised by *feature generator* rather than by regime because the
    DCQF transform depends only on X and never on y: the three uniform regimes share
    one X per seed, so their fitted extractors and feature matrices are identical and
    computing them once instead of three times pays for the seed increase outright.
    """
    per_fold = {r: {k: [] for k in ARMS} for r in REGIMES}
    per_seed = {r: {k: [] for k in ARMS} for r in REGIMES}
    spike_fold = {r: {e: [] for e in SPIKES} for r in REGIMES}
    spike_seed = {r: {e: [] for e in SPIKES} for r in REGIMES}
    gens = {"uniform": make_uniform, "cardio": make_cardio_like}

    for gen_name, gen in gens.items():
        regimes = [r for r, g in REGIMES.items() if g == gen_name]
        for seed in range(n_seeds):
            X = gen(n, np.random.default_rng([2026, seed]))
            # A separate label stream per regime, so adding or removing a regime
            # cannot shift another one's labels through a shared generator state.
            labels = {r: make_labels(r, X, np.random.default_rng([77, seed, i]))
                      for i, r in enumerate(regimes)}
            acc = {r: {k: [] for k in ARMS} for r in regimes}
            sacc = {r: {e: [] for e in SPIKES} for r in regimes}

            idx = np.random.default_rng(seed).permutation(n)
            for f in range(folds):
                te = idx[f::folds]
                tr = np.setdiff1d(idx, te)
                Xtr, Xte = to_angles(X[tr], X[te])

                B = {}
                for nm, tf in (("dcqf", DCQFExtractor(orders_encoded=(2,))),
                               ("scr", DCQFExtractor(orders_encoded=(2,),
                                                     scramble_couplings=True)),
                               ("prod", ChainProductControl())):
                    tf.fit(Xtr)
                    B[nm] = (tf.transform(Xtr), tf.transform(Xte))

                def cat(*keys):
                    return (np.hstack([Xtr] + [B[k][0] for k in keys]),
                            np.hstack([Xte] + [B[k][1] for k in keys]))

                arms = {"raw X (8)": (Xtr, Xte),
                        "DCQF (24)": B["dcqf"],
                        "DCQF scrambled (24)": B["scr"],
                        "chain products (24)": B["prod"],
                        "raw+DCQF (32)": cat("dcqf"),
                        "raw+scrambled (32)": cat("scr"),
                        "raw+products (32)": cat("prod")}

                for r in regimes:
                    y, s = labels[r]
                    ytr, yte = y[tr], y[te]
                    for k, (a, b) in arms.items():
                        v = score(a, ytr, b, yte)
                        per_fold[r][k].append(v)
                        acc[r][k].append(v)
                    for e in SPIKES:
                        a, b = spike(B["dcqf"][0], B["dcqf"][1], s[tr], s[te], e)
                        v = score(a, ytr, b, yte)
                        spike_fold[r][e].append(v)
                        sacc[r][e].append(v)

            for r in regimes:
                for k in ARMS:
                    per_seed[r][k].append(float(np.mean(acc[r][k])))
                for e in SPIKES:
                    spike_seed[r][e].append(float(np.mean(sacc[r][e])))

    return {"per_fold": per_fold, "per_seed": per_seed,
            "spike_fold": spike_fold, "spike_seed": spike_seed}


def report(raw: dict, n: int, n_seeds: int, folds: int) -> dict:
    out = {}
    for r in REGIMES:
        pf, ps = raw["per_fold"][r], raw["per_seed"][r]
        base_f, base_s = np.array(pf[REFERENCE]), np.array(ps[REFERENCE])

        print(f"\n=== {r} ===  {n_seeds} seeds x {folds} folds, n={n}")
        print(f"  {'arm':<21} {'AUC':>7} {'sd':>6}  {'DCQF-this':>10} {'t/fold':>7} "
              f"{'t/seed':>7} {'p':>8}  {'90% CI':>18} {'excl.>':>7}")
        rec = {}
        for k in ARMS:
            v = np.array(pf[k])
            row = {"auc": float(v.mean()), "sd": float(v.std())}
            line = f"  {k:<21} {v.mean():>7.4f} {v.std():>6.4f}"
            if k != REFERENCE:
                d_seed = base_s - np.array(ps[k])
                t = paired_t_test(d_seed)
                eq = tost_equivalence(d_seed, margin=MARGIN)
                tf_ = paired_t_test(base_f - v).t
                row.update({"delta": float(base_f.mean() - v.mean()),
                            "t_fold": float(tf_), "t_seed": float(t.t),
                            "p_seed": float(t.p_value),
                            "ci_low": float(t.ci_low), "ci_high": float(t.ci_high),
                            "significant": bool(t.p_value < 0.05),
                            "equiv_bound": float(eq.bound),
                            "equivalent_at_margin": bool(eq.equivalent)})
                line += (f"  {row['delta']:>+10.4f} {tf_:>+7.2f} {t.t:>+7.2f} "
                         f"{t.p_value:>8.4f}  "
                         f"[{t.ci_low:>+7.4f},{t.ci_high:>+7.4f}] {eq.bound:>7.4f}")
            print(line)
            rec[k] = row

        # ---- the positive control
        print(f"\n  spiked positive control ({r}) -- DCQF with a known signal mixed in")
        print(f"  {'eps':>5} {'AUC':>8} {'lift':>9} {'t/seed':>8} {'p':>9}  detected")
        sp = {}
        b_s = np.array(raw["spike_seed"][r][0.0])
        b_f = np.array(raw["spike_fold"][r][0.0])
        for e in SPIKES:
            v_f = np.array(raw["spike_fold"][r][e])
            v_s = np.array(raw["spike_seed"][r][e])
            if e == 0.0:
                print(f"  {e:>5.2f} {v_f.mean():>8.4f} {'--':>9} {'--':>8} {'--':>9}"
                      f"  (identity check)")
                sp[str(e)] = {"auc": float(v_f.mean()), "lift": 0.0}
                continue
            t = paired_t_test(v_s - b_s)
            det = t.p_value < 0.05 and t.mean > 0
            sp[str(e)] = {"auc": float(v_f.mean()),
                          "lift": float(v_f.mean() - b_f.mean()),
                          "t_seed": float(t.t), "p_seed": float(t.p_value),
                          "detected": bool(det)}
            print(f"  {e:>5.2f} {v_f.mean():>8.4f} {v_f.mean() - b_f.mean():>+9.4f} "
                  f"{t.t:>+8.2f} {t.p_value:>9.5f}  {'YES' if det else 'no'}")

        detected = [sp[str(e)] for e in SPIKES
                    if e > 0 and sp[str(e)].get("detected")]
        mde = min((d["lift"] for d in detected), default=float("nan"))
        print(f"  -> smallest detected lift (empirical MDE): {mde:+.4f} AUC")
        out[r] = {"arms": rec, "spike": sp, "mde": float(mde)}
    return out


def calibrate(n: int, seeds: int) -> None:
    """Print the achievable AUC per regime so the noise settings can be checked.

    The cardio-like regime is meant to reproduce the ~0.79 AUC ceiling measured on
    the real cardio_train. If this drifts, NOISE['cardio-like'] needs re-tuning and
    every claim about that regime being realistic is void.
    """
    for r, g in REGIMES.items():
        gen = {"uniform": make_uniform, "cardio": make_cardio_like}[g]
        vals = []
        for seed in range(seeds):
            X = gen(n, np.random.default_rng([2026, seed]))
            y, _ = make_labels(r, X, np.random.default_rng([77, seed, 0]))
            cut = int(0.8 * n)
            Xtr, Xte = to_angles(X[:cut], X[cut:])
            vals.append(score(Xtr, y[:cut], Xte, y[cut:]))
        print(f"  {r:<12} noise={NOISE[r]:<5} logreg AUC = {np.mean(vals):.4f} "
              f"+- {np.std(vals):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--calibrate", action="store_true",
                    help="print the achievable AUC per regime and exit")
    ap.add_argument("--out", type=str, default="results/runs/dcqf_null_test.json")
    a = ap.parse_args()

    if a.calibrate:
        calibrate(a.n, a.seeds)
        return 0

    t0 = time.time()
    raw = collect(a.n, a.seeds, a.folds)
    res = report(raw, a.n, a.seeds, a.folds)
    dt = time.time() - t0

    df = a.seeds - 1
    print(f"\n[{dt:.0f}s]  Quote 't/seed': {a.seeds} independent dataset draws, "
          f"{df} df. 't/fold' is anti-conservative (folds share training rows).")
    print(f"'excl.>' is the smallest effect the data rule out at 95% -- the number "
          f"to quote for a null, not the p-value.")

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"config": vars(a), "seconds": dt, "margin": MARGIN,
                             "spikes": list(SPIKES), "noise": NOISE,
                             "results": res}, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
