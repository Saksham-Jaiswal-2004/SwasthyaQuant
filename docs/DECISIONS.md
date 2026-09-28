# Decision log

Each entry records what was decided, what was rejected, and what would make us revisit
it. Dates are absolute.

---

## ADR-001 — Tabular heart disease, not breast cancer imaging or genomics
**2026-08-30 · accepted**

The earlier roadmap targeted the Wisconsin breast cancer dataset. Rescoped to coronary
heart disease at the team's request.

**Why it works better than expected.** The quantum kernel's O(N²) cost makes small
datasets a feature rather than an embarrassment, and clinical tabular data has named
features, so SHAP output reads as "downsloping ST raised risk" instead of "component 3
increased". A cardiologist can argue with the first kind of explanation, which is the
point of explainability.

**Rejected.** Chest X-ray (a CNN front-end would dominate, and the quantum layer would
be decorative on 96 parameters against 25M). Genomics (thousands of features, no
plausible route to 4 qubits without a reduction step that does all the work).

**Revisit if** a judge specifically asks for imaging. Chest X-ray is Horizon 2 in the
roadmap, deliberately unbuilt.

---

## ADR-002 — Kaggle Heart Failure Prediction (918 rows) as the primary source
**2026-08-30 · accepted, with a known cost**

**Why.** One clean file, 918 de-duplicated rows, 11 features, widely used so the
baselines are checkable against published work.

**The cost, stated plainly.** The merge that produced this file **dropped the
source-hospital column.** Cross-cohort external validation is therefore impossible on
it. This is a real loss: leave-one-hospital-out is the strongest generalisation evidence
available for this data, and we cannot produce it from the primary source.

**How the code handles it.** `cohort_holdout()` raises with an explicit message rather
than silently substituting a random split. A function that quietly does something weaker
than its name promises is worse than one that fails.

**Revisit if** there is time in Week 3. `configs/data/heart_uci_cohorts.yaml` reads the
four raw UCI files, keeps both `severity` and `cohort`, and enables the real thing.

---

## ADR-003 — Subgroup analysis as a partial substitute for external validation
**2026-08-30 · accepted**

Given ADR-002, generalisation evidence comes from subgroup metrics by sex and age band.

**Why these axes.** The dataset is ~79% male, and cardiology models underperforming on
women is a documented, consequential failure mode. Age is the dominant risk gradient, so
a model could be riding age alone.

**What this is not.** It is not external validation. It measures whether performance
holds across groups *within one distribution*; it says nothing about a new hospital.
`eval/subgroups.py` states this in its module docstring, `data/audit.py` states it in
its warnings, and the README states it under limitations. Three places, because this is
the claim most likely to get overstated under pressure.

**Guard.** Subgroups with n < 30 are returned with `reportable=False` and drawn hatched
rather than dropped. A disparity that disappears because the group was small is the
disparity most worth seeing.

---

## ADR-004 — Binary target now; severity kept as a nullable column
**2026-08-30 · accepted**

The team wanted binary classification now with the severity extension left open. The
Kaggle merge ships only a binary `HeartDisease` column, so these two wishes conflict.

**Resolution.** `schema` declares `severity` and `cohort` as **nullable by design**. The
Kaggle loader sets them to NaN; the UCI loader fills them from `num` (0–4) and the source
file. `schema.validate(require_severity=True)` raises with a message naming the loader to
switch to. Nothing downstream needs restructuring when the extension happens.

---

## ADR-005 — PennyLane for models, Qiskit for hardware and noise
**2026-08-30 · accepted**

PennyLane's autograd interface backpropagates through the simulator, which trains a VQC
far faster than parameter-shift. Qiskit Aer has the better noise models and is the route
to IBM Runtime.

**The honest caveat, and the answer to give.** Backprop through a statevector simulator
is not physically available on hardware. We therefore run the VQC once with
`gradient_method="parameter-shift"` so the hardware-honest cost is *measured* rather than
asserted: 97 parameters × 2 evaluations = 194 circuit evaluations per step per sample.

---

## ADR-006 — Fixed clinical angle bounds instead of a fitted MinMaxScaler
**2026-08-30 · accepted — this is what makes the project affordable**

A quantum kernel value depends on no labels, so in principle the full 918×918 Gram
matrix can be computed once and sliced per fold: **11 hours becomes 28 minutes.** The
obstacle is the scaler in front of it. A `MinMaxScaler` fitted on all 918 rows puts the
test fold's min and max into the training representation — no labels leak, but feature
statistics do, and a careful reviewer is entitled to call that contamination.

**Resolution.** Scale to [0, π] using `schema.PLAUSIBLE` — physiological bounds, chosen
before seeing the data. The transform has no fitted state, so the global Gram cache is
*provably* leak-free.

**The price, stated.** Fixed bounds waste dynamic range when observed spread is narrower
than the physiological range (cholesterol could span 80–700 but clusters near 200), which
compresses the angles slightly. Cheaper than either eleven hours or a contaminated
benchmark.

**Revisit if** anyone swaps in a fitted scaler. `configs/model/qkernel_zz.yaml` records
the dependency: switch the scaler and you must disable the cache.

---

## ADR-007 — Control-C runs both brackets, not one
**2026-08-30 · accepted**

Integer hidden widths rarely hit the circuit's parameter count exactly. For 96 angles
with d_in=8 and d_out=4, `13h + 4 = 96` gives h = 7.08 — so h=7 yields 95 parameters and
h=8 yields 108.

**Decision.** Run both and report exact counts. Reporting only the smaller control would
be the same selective-comparison error the reference papers make, just pointed the other
way.

**What we do if the result straddles.** If the quantum model beats the 95-parameter
control and loses to the 108-parameter one, the honest conclusion is that the comparison
is a wash, and we say so. That sentence is what makes the rest of the table credible.

---

## ADR-008 — The default extractor is a *fixed random* circuit
**2026-08-30 · accepted**

`QuantumExtractor(trainable=False)` by default.

**Why this is stronger, not lazier.** A fixed random quantum feature map versus a fixed
random classical projection of equal size holds training constant at zero on both sides.
The only remaining difference is the nature of the feature space, which is the actual
question. Adding training to both sides adds a confound — optimiser behaviour — to a
comparison that does not need one.

Trained versions of both are run as a second condition. Both go in the table.

---

## ADR-009 — Pure numpy for metrics and splits
**2026-08-30 · accepted**

Metrics and splits are implemented from scratch: ROC-AUC via Mann-Whitney U, average
precision with sklearn's step definition, MCC, kappa, Brier, ECE, and the fold
generator.

**Why.** The numbers we defend should not move because a library was upgraded between
the rehearsal and the final run. It also makes the correctness-critical core testable
under bare Python plus numpy, and it forced us to know exactly what each metric does —
which is the difference between reporting PR-AUC and understanding why it beats ROC-AUC
under imbalance.

**Verified against** hand-derived values with the derivation written into the test
comments, not against scikit-learn — a test that compares two implementations of the same
misunderstanding passes.

---

## ADR-010 — Entangling gates are excluded from parameter counts
**2026-08-30 · accepted**

CNOT has no continuous parameter to learn. Reference paper P1 counts its entanglers,
inflating 16 angles per layer to 19 and 96 total to 114.

`count_circuit_params` returns entangler count as a separate, labelled field so it can be
reported without being summed. `ansatz.assert_param_count` cross-checks the weight tensor
against the analytic count on every fit — the failure this catches is a reshape bug that
leaves the model training happily on 64 parameters while the report claims 96.
