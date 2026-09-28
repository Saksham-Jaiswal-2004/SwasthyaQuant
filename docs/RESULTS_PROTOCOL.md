# Results protocol

**Written 2026-08-30, before any model was fitted.** That is the entire point of the
document: a rule about what counts as a result is only meaningful if it was written
before the results existed. Anything added after the first table lands is marked with its
date and a reason.

## What we will claim

1. A variational circuit with **96 trainable angles (384 bytes at fp32)** achieves
   sensitivity and specificity within a stated margin of classical baselines that use
   between 100 and 1,200 parameters.
2. That comparison is measured against a **parameter-matched classical control**, with
   exact parameter counts in a labelled column.
3. The benchmark harness is fold-safe, seeded, and reproducible from a single command.
4. Performance under finite sampling degrades along a measured curve, not a single point.

## What we will not claim

- **Quantum advantage in accuracy.** We expect XGBoost to win the accuracy column and we
  put that on our own slide.
- **Clinical utility.** No model here is validated for clinical use. Not a medical device.
- **Generalisation to new hospitals.** The primary dataset makes that untestable
  (`docs/DECISIONS.md`, ADR-002). Subgroup analysis is not external validation.
- **Hardware advantage.** One IBM Runtime job is supporting evidence for feasibility, not
  a performance claim. A simulator-produced figure is never labelled "hardware".

## Metric hierarchy, fixed in advance

Headline, always present, enforced by `report.tables.refuse_if_incomplete`:
**sensitivity** and **specificity**. In screening, a missed case is the costly error, and
accuracy alone hides it.

Primary discriminative metric: **PR-AUC**. Not ROC-AUC, which flatters a model under
class imbalance. Both are reported; PR-AUC is the one we argue from.

Also reported: PPV, NPV, balanced accuracy, MCC, Cohen's kappa, Brier score, expected
calibration error, and accuracy — last, and always quoted against the majority-class
floor recorded by the audit.

## Validation protocol

Five repeats of stratified 5-fold, so 25 fits per model, mean ± sd. **No single-split
number is ever reported.** On ~184 test rows, a one-point difference is roughly two
patients, which is well inside fold-to-fold noise.

Standard deviation across folds is reported as a **spread, not a confidence interval**.
Folds share training data, so it understates true uncertainty. Bootstrap CIs are computed
on the sealed holdout where an interval is needed.

The decision threshold is chosen on **training folds only**, by Youden's J. Tuning a
threshold on test can move sensitivity several points and is almost never disclosed.

A sealed 20% holdout is opened **once**, in Week 4, after all model selection is
finalised. If we open it twice, it stops being a holdout and every number derived from it
is a training number.

## Statistical comparison

Paired comparisons on out-of-fold predictions use **McNemar's test** — exact binomial
under 25 discordant pairs, Edwards continuity-corrected chi-square above. Below 10
discordant pairs the verdict is reported as **inconclusive**, not as a null result.

Metric differences across folds use the **Nadeau–Bengio corrected resampled t-test**,
which inflates the variance to account for overlapping training sets. The uncorrected
version is anti-conservative on repeated CV and produces significance that is not there.

Multiple comparisons are corrected with **Holm–Bonferroni**. Five comparisons at α = 0.05
give roughly a 23% chance of at least one spurious hit; that is too high to leave alone
in a table people will read as evidence.

## Reporting rules the code enforces

A table without sensitivity and specificity raises. A quantum model reported without a
`control_c` row raises. A control whose parameter count has drifted more than 15% from
the circuit's raises. These are exceptions rather than documentation because the moment
they matter is the moment nobody has time to check a README.

## Failure modes we will report rather than hide

**Barren plateau.** If VQC loss does not move across training, that is the barren-plateau
signature, not a hard problem. `VQC._flat_loss_warning` detects it. The resulting
chance-level accuracy is never reported as a quantum result.

**Provenance leakage.** If a missingness indicator ranks in the top three SHAP features,
the model may be classifying source hospital rather than physiology.
`explain.check_leakage_proxies` flags it, and we then report both numbers — with and
without the indicators.

**Label noise ceiling.** The audit reports feature-identical rows with disagreeing labels.
That sets a hard ceiling on achievable accuracy, and knowing it is what stops the team
chasing 99% on a dataset that cannot support it.

**Subgroup disparity.** If sensitivity on women is materially below sensitivity on men, it
goes in the report. A model that works well on 79% of patients and poorly on 21% is a
finding, not a rounding error.

## The sentence we are aiming for

> On 918 rows of clinical data, a 4-qubit variational circuit with 96 trainable
> parameters reached sensitivity X ± Y, against Z ± W for a classical control with 95
> parameters and the same classifier head. Gradient boosting, with roughly 10,000
> parameters, reached V ± U. We report the parameter-matched comparison because it is the
> only one that isolates the contribution of the quantum feature space, and it is the
> comparison both of the papers underlying this problem statement omit.

Every number in that sentence traces to a row in `results/runs/ledger.csv`.
