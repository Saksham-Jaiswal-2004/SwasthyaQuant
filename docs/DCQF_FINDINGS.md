# DCQF: what we measured, and why it is a result rather than a disappointment

**Status:** complete, reproducible via `PYTHONPATH=src python scripts/dcqf_null_test.py`
**Raw output:** `results/runs/dcqf_null_test.json`
**Date:** 2026-09-09

---

## The one-sentence finding

We implemented Kipu Quantum's digitized counterdiabatic quantum feature extraction
faithfully, verified every gate against dense matrix exponentials, and then ran the
experiment the paper omits — and **DCQF does not outperform its own null model on any
label regime we tested**. Its measurable advantage over raw features comes from
expanding 8 columns into 24, not from the mutual-information encoding that is the
paper's actual contribution.

This is a finding worth presenting. It is also exactly what the project positioning
committed to: *rigour, not accuracy supremacy*.

---

## What was being tested

DCQF makes two specific claims beyond "we did something quantum". Eq. (2) says the
couplings of the spin-glass Hamiltonian should carry the **mutual information between
feature pairs**, so the encoding is informed by the data's correlation structure. The
counterdiabatic evolution then says the **quantum dynamics** produce features a
classical map would not.

Neither claim is tested by comparing DCQF against raw features, because DCQF also turns
8 features into 24 and more columns fit more label. So each DCQF arm was run against
two comparators on byte-identical folds with a byte-identical classifier head:

| Comparator | What it holds fixed | What it destroys | Isolates |
|---|---|---|---|
| **Scrambled DCQF** | the whole circuit, the coupling magnitudes, the output width | which variable each coupling attaches to | Eq. (2), the MI encoding |
| **Chain products** | the subsets measured, the output width, the output range | the circuit entirely | the quantum step |

The chain-product control multiplies features over the same closed-chain subsets the
circuit measures, then applies `tanh` so both arms hand the head inputs in the same
`[-1, 1]` range. Without the squash, an unbounded control against a bounded treatment
is a scaling artefact waiting to be reported as a finding.

Three label regimes, because "DCQF works" without "on what" is how the literature got
here. `linear` is where `cardio_train` actually lives — a single blood-pressure
threshold already reaches 71.5% on it. `pairwise` is built to suit DCQF: pure
`cos(x_i)cos(x_j)` structure with no linear signal at all. `mixed` is the realistic
clinical case.

---

## Results

Four independent dataset draws × five folds, n=400, 35% label noise. AUC from one
shared logistic head; standardisation fitted on training rows only.

### The comparison that matters: DCQF minus its scrambled null

| Regime | DCQF | Scrambled | Difference | t (per seed, 3 df) | Significant |
|---|---|---|---|---|---|
| linear | 0.7605 | 0.7744 | −0.0139 | −1.60 | no |
| pairwise | 0.6410 | 0.6275 | +0.0135 | +0.92 | no |
| mixed | 0.7653 | 0.7733 | −0.0080 | −1.33 | no |

**Not once, in any regime, including the one built to favour it.** The critical value
at 3 degrees of freedom is 3.18; the largest observed statistic is 1.60, and it points
the *wrong way*. Permuting the couplings off their variables costs nothing — which
means the mutual-information structure of Eq. (2) is not reaching the output.

### DCQF versus the classical product control

| Regime | DCQF | Chain products | Difference | t (per seed) | Significant |
|---|---|---|---|---|---|
| linear | 0.7605 | 0.8978 | −0.1373 | −5.45 | yes, **against** DCQF |
| pairwise | 0.6410 | 0.6352 | +0.0058 | +0.30 | no |
| mixed | 0.7653 | 0.8767 | −0.1114 | −4.06 | yes, **against** DCQF |

On its best case DCQF ties a plain product. On the two realistic ones it loses
decisively to arithmetic.

### DCQF versus raw features — where the apparent win comes from

| Regime | Raw (8 cols) | DCQF (24 cols) | Difference | t (per seed) |
|---|---|---|---|---|
| linear | 0.9610 | 0.7605 | **−0.2005** | −9.11 |
| pairwise | 0.4969 | 0.6410 | **+0.1440** | +5.55 |
| mixed | 0.9443 | 0.7653 | **−0.1791** | −7.66 |

DCQF beats raw features only in the regime with no linear signal — and there the
scrambled null and the product control match it. On linear and mixed labels it
**destroys** information: bounded expectation values compress a monotone relationship
that logistic regression was reading directly.

That last row is the one with teeth for this project. `cardio_train` is near-linear in
blood pressure. The `mixed` regime is the closest analogue in this experiment, and DCQF
costs 0.18 AUC there.

### The paper's headline configuration

The paper's best results come from "Hybrid" — classical features concatenated with
quantum ones. Reproduced here, that arm never beats raw features alone:

| Regime | Raw | Raw+DCQF | Raw+scrambled | Raw+products |
|---|---|---|---|---|
| linear | 0.9610 | 0.9516 | 0.9516 | 0.9553 |
| pairwise | 0.4969 | 0.6430 | 0.6298 | 0.6216 |
| mixed | 0.9443 | 0.9393 | 0.9379 | 0.9412 |

Adding DCQF columns to raw features changes nothing a scrambled version would not also
change. Note `raw+DCQF` ≈ `raw+scrambled` in all three rows.

---

## A defect in the paper's own headline number

Independent of our replication, the paper's largest claimed gain rests on a
methodological error that inflates it, and this should be raised if DCQF is discussed
at judging.

The paper states, verbatim:

> "The combined set is then ranked by SHAP importance derived from a Gradient Boosting
> classifier **trained on the full set**, and the top-K features are retained. […] All
> cross-validated metrics reported below for the Hybrid configuration use exclusively
> these retained features as input to the Gradient Boosting classifier."

Feature selection is performed once, on all the data, and the cross-validation is then
run on the surviving features. Every test fold in that CV was seen by the selector.
This is textbook **selection leakage**, and the correct procedure — selecting inside
each training fold — is standard and would have cost the authors nothing but compute.

The magnitude is not small. On the molecular toxicity dataset the reported jump is
F1-macro **0.515 ± 0.083 → 0.727 ± 0.091**, on n=171 with 156 descriptors. Selecting
50 of 156+ features on 171 samples using the labels is precisely the configuration in
which selection leakage produces the largest optimistic bias. The paper's own
uncontaminated numbers tell a much flatter story:

| Dataset | Best standalone quantum | Classical original | Winner |
|---|---|---|---|
| Molecular toxicity (n=171) | 0.577 ± 0.091 (2-body) | 0.515 ± 0.083 | quantum, within error bars |
| Breast MedMNIST (n=702) | 0.758 ± 0.041 (3-body) | 0.778 ± 0.033 | **classical** |

On the larger of the two datasets, every standalone quantum feature set loses to the
plain classical features. Only the leaky Hybrid configuration wins.

One further concern, stated as a question because the paper does not address it
directly: Table 2 reports MedMNIST predefined-test-split results for "SHAP-selected"
features. The paper describes that dataset as "702 samples including both training and
test sets" and describes the selection as trained on "the full set". If the selection
saw the predefined test split, the Table 2 comparison against ResNet18/ResNet50/Google
AutoML is not like-for-like either. The paper does not restate the protocol there, so
this is unresolved rather than established.

Cite the paper as **accepted-article-in-press** (doi:10.1038/s41598-026-67564-0); the
authors note the text may still change, so re-check these quotations against the final
version.

---

## What this does and does not establish

**Established.** Within this experiment — synthetic data, 8 features, 4 seeds, one
logistic head, orders (2,) encoded and (1,2,3) read out — the MI coupling encoding
contributes nothing detectable over a magnitude-matched permutation, and the whole
DCQF map is matched or beaten by classical products of the same subsets.

**Not established.** That DCQF is worthless. Four seeds is 3 degrees of freedom, which
can only detect large effects; a real effect of ~0.01 AUC would be invisible here. The
test is on synthetic data, not `cardio_train`. Higher encoded orders, a different
readout set, a stronger head, or a genuinely high-dimensional problem (the paper's 156
descriptors, not our 8) could all change the answer. The honest claim is *"we could not
detect a contribution from the quantum-specific components at this scale"*, not
*"there is none"*.

**The natural next step**, if this is pursued: run the same three-arm comparison on the
real `cardio_train` features rather than synthetic ones, and raise the seed count so
the per-seed test has power to detect small effects.

---

## For the SIH slide

Do not put "our quantum method wins" on a slide. Put this:

> We implemented a 2026 *Scientific Reports* quantum feature-extraction method, verified
> every gate against exact matrix exponentials, and tested it against two controls the
> original paper does not run: a scrambled null and a dimension-matched classical
> equivalent. It beats neither. We also identified selection leakage in the paper's
> headline result. **Our platform is built to detect this — including in our own work.**

That is a stronger claim than a marginal accuracy win, and unlike a marginal accuracy
win, it survives a judge who knows statistics. The DCQF arm ships in the repo as
`dcqf`, and the registry refuses to let it be reported without `dcqf_scrambled` and
`dcqf_product_control` alongside it.

---

## Reproducing

```bash
PYTHONPATH=src python scripts/dcqf_null_test.py --seeds 4 --n 400
```

~96 s. Writes `results/runs/dcqf_null_test.json`. Every gate used is covered by
`tests/test_quantum_circuits.py`, which checks the circuit against dense matrix
exponentials (worst error 8.6e-16 across 9 configurations), the parameter-shift
gradient against central finite differences (3.3e-11, at a point where the gradient
scale is 0.95 so the check cannot pass vacuously), and includes a tripwire for the
failure mode where the counterdiabatic term silently vanishes and every feature
becomes exactly zero.
