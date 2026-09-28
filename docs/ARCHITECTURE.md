# Architecture

## How data moves

```
data/raw/heart.csv
        |
        v
  data/loaders.py          renames, maps categories, converts sentinels to NaN.
        |                  Does NOT impute, scale, or drop rows.
        v
  schema.validate()        contract check. Raises on an unlisted category, because an
        |                  unmapped code usually means a column shifted.
        v
  data/audit.py            class balance, sentinel counts, duplicates, contradictory
        |                  rows and the label-noise accuracy ceiling. Run before models.
        v
  preprocess.encode_frame  stateless one-hot from the schema's category lists, plus
        |                  missingness indicators. Cannot leak: nothing is fitted.
        v
  features/select.py       eight named clinical columns (default) or PCA-8.
        |
        v
  eval/splits.py           5 repeats x stratified 5-fold. Every fold self-checked.
        |
        +--> for each fold:  Pipeline([preprocessor, model]).fit(train)
        |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ imputer and scaler are
        |                    fitted HERE, on training rows only. This is the whole
        |                    reason preprocessing is a pipeline step and not a
        |                    preparatory script.
        v
  eval/harness.py           threshold chosen on the TRAINING fold, applied to test.
        |                   Out-of-fold predictions retained for McNemar.
        v
  results/runs/ledger.csv   append-only, one row per (model, repeat, fold)
        |
        v
  report/tables.py          aggregation + the gate that refuses an incomplete table
  report/figures.py         six figures, greyscale-readable
```

## Why each boundary exists

**Loader / preprocessor.** A loader that imputes is a loader that leaks. Imputation
needs a statistic, that statistic must come from training rows only, and the loader runs
once before any split exists. So the loader converts impossible values to NaN and stops
there. Splitting these two jobs is the single structural decision that makes fold safety
achievable rather than a matter of discipline.

**Encode / preprocess.** `encode_frame` is stateless because it takes its category lists
from `schema.CATEGORIES`, not from the data. If categories were discovered from the data,
the column layout would depend on which rows you happened to see — so a fold missing one
rare chest-pain type would produce a narrower matrix, and the model would train happily
on misaligned columns.

**Harness / model.** The harness knows only `fit` and `predict_proba`. It never branches
on model type. That is what makes the comparison apples-to-apples: the most common way
benchmark papers go wrong is evaluating the new method through a different code path
from the baselines, usually with slightly kinder rules.

**Model / report.** The report layer reads the ledger, not the model objects. So every
number in the final table traces to an append-only row that records the model, the
config, and the fold it came from.

## The one special case

The quantum kernel does not consume a feature matrix. An SVC over a precomputed Gram
matrix consumes **row indices** into the cached kernel. Rather than scattering
`if name == "qkernel"` through the CLI, `ModelSpec.input_kind` declares it in the
registry, in one readable place.

## Design rules, and the failure each one prevents

**1. Config-driven, single entry point.** Everything runs through
`python -m qheart.cli`. *Prevents:* the 2 a.m. situation where a number in the deck came
from a notebook cell nobody can find, run in an order nobody remembers.

**2. Fold-safe preprocessing, enforced structurally.** The preprocessor is a pipeline
step. *Prevents:* leakage that inflates every downstream number by a point or two,
invisibly, because the result looks *better* and so never triggers suspicion.

**3. One model interface.** `fit` / `predict_proba`, registry-built factories.
*Prevents:* the new method being scored under quieter rules than the baselines.

**4. Parameter accounting as a first-class module.** `params.py` has its own tests, and
Control-C is sized from the live circuit. *Prevents:* the headline claim drifting out of
match after an unrelated config change, while every number still looks plausible.

**5. Append-only results ledger.** One row per model per fold, never overwritten.
*Prevents:* the quiet deletion of a disappointing run, which biases the report far more
than any modelling choice.

**6. Cost guards before commitment.** `guard_kernel_size` raises; `estimate_kernel_cost`
prints. *Prevents:* discovering the O(N²) wall by watching a progress bar for two days
during the last week before submission.

## Seeding

`seeds.py` is the only module that seeds anything. `derive()` uses SHA-256 rather than
Python's `hash()`, because `hash()` is salted per process and would silently break
reproducibility across runs. Every purpose gets its own derived stream — splits, ansatz
initialisation, batch order, noise repeats — so changing the number of epochs does not
change the folds.

## Dependency layering

`requirements.txt` holds core scientific Python only. Quantum dependencies live in
`requirements-quantum.txt`, and **every** import of pennylane, qiskit, torch or shap in
this codebase is inside a function.

This is not stylistic. It means a broken `pip install qiskit-aer` — which will happen at
least once, probably the week before submission — cannot stop anyone from producing the
baseline table. It also means `schema`, `splits`, `metrics` and `params` run under bare
Python plus numpy, which is how they are unit-tested.

Metrics and splits are implemented in pure numpy rather than delegating to scikit-learn.
The reason is narrow and practical: the numbers we defend at judging should not move
because someone upgraded a library between the rehearsal and the final run.
