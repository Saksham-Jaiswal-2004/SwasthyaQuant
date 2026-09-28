# Data card — Heart Failure Prediction

## Provenance

Primary source: **Heart Failure Prediction Dataset**, Kaggle (fedesoriano), 918 rows and
11 features plus a binary target. It is a merge of five cohorts from the UCI Machine
Learning Repository — Cleveland, Hungarian, Switzerland, Long Beach VA, and Stalog — with
duplicate patients removed, reducing 1,190 rows to 918.

Original UCI donors: Andras Janosi (Hungarian Institute of Cardiology), William Steinbrunn
(University Hospital Zurich), Matthias Pfisterer (University Hospital Basel), and Robert
Detrano (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation).

Licence: the UCI Heart Disease datasets are released for research use with attribution.
Confirm the Kaggle page's licence field before publication, and cite the UCI donors rather
than only the Kaggle mirror.

Not redistributed in this repository. `scripts/fetch_data.sh` downloads it; `.gitignore`
excludes `data/raw/`.

## Columns

Eleven features: age, sex, chest pain type, resting blood pressure, cholesterol, fasting
blood sugar, resting ECG, maximum heart rate achieved, exercise-induced angina, ST
depression (oldpeak), and ST slope. Target is `HeartDisease`, renamed to `target`.

Canonical names, types, categories and plausible ranges are defined in
`src/qheart/schema.py`. Run `python -m qheart.cli schema` to print the contract.

## Known defects

Run `make audit` for exact counts on your copy. Every item below is something that would
otherwise become a wrong number in the report.

**Sentinel zeros.** Roughly 172 rows have `cholesterol == 0` and at least one has
`RestingBP == 0`. Zero means "not measured", not a measurement of zero. Treated as NaN,
never clipped. Critically, the pattern of missingness partly encodes **which hospital the
row came from** — Switzerland has almost no cholesterol measurements — so missingness is
informative, and a model can score well by identifying the hospital instead of the
patient. Missingness indicators are added deliberately, and
`explain.check_leakage_proxies` flags it if they become top-ranked features.

**No cohort column.** The merge dropped source hospital, so cross-hospital external
validation is impossible on this file. `cohort_holdout()` raises rather than substituting
a random split. See ADR-002 and ADR-003.

**Sex imbalance.** Approximately 79% male. Sensitivity is reported per sex; subgroups with
fewer than 30 patients are marked not reportable and drawn hatched rather than dropped.

**Binary target only.** UCI's ordinal severity (`num`, 0–4) did not survive the merge.
`schema` keeps `severity` as a nullable column; `configs/data/heart_uci_cohorts.yaml`
recovers it from the raw files.

**Contradictory rows.** Feature-identical rows with disagreeing labels appear after
de-duplication on a reduced feature set. The audit counts them and reports a
`label_noise_ceiling_hint` — the maximum accuracy any model could reach. Knowing it is
what stops the team chasing 99%.

**Class balance.** Roughly 55% positive, so the majority-class floor is about 0.55. Any
accuracy figure is quoted against that floor; the audit records it as
`majority_class_accuracy`.

## Intended and unintended use

Intended: benchmarking model architectures under a fixed, fold-safe evaluation protocol
for a hackathon submission.

**Not intended:** clinical decision support, risk scoring for real patients, or any use
implying validation. The cohorts are decades old, drawn from four countries, with unknown
selection criteria and no follow-up. Prevalence in this data does not reflect any
screening population.

## Preprocessing applied

The loader renames columns to schema names, maps category codes to readable labels,
converts sentinel zeros and physiologically impossible values to NaN, and records counts
in `df.attrs`. It does not impute, scale, or drop rows — a loader that imputes is a loader
that leaks. Imputation and scaling happen inside each cross-validation fold.
