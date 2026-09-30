# Swasthya Quant

**Hybrid Quantum Intelligence for Early Disease Detection**

Smart India Hackathon 2026 · Problem Statement **26139**: *Hybrid Quantum Machine Learning Platform for Early Disease Detection*

Swasthya Quant is a hybrid quantum-classical machine learning platform for estimating cardiovascular disease risk. A quantum feature map, digitized counterdiabatic quantum feature extraction (DCQF), adds quantum-derived features to routine clinical measurements, and a classical classifier turns them into a risk probability. Every model, quantum or classical, is scored by one fold-safe evaluation harness and compared against the classical controls that could explain its result.

> **Not a medical device.** Every output is a model-estimated risk for research and demonstration. It is not a diagnosis, and no model here has been validated for clinical use.

---

## Contents

- [Project status](#project-status)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [API reference](#api-reference)
- [Web application](#web-application)
- [Research pipeline](#research-pipeline)
- [Results so far](#results-so-far)
- [Testing](#testing)
- [Known issues](#known-issues)
- [Demo flow](#demo-flow-2-3-minutes)
- [Scope and limitations](#scope-and-limitations)
- [References](#references)

---

## Project status

| Component | State |
|---|---|
| Research package `src/qheart` (data, features, quantum circuits, evaluation) | Implemented, with unit tests |
| Classical baselines (logreg, SVM-RBF, RF, XGBoost, MLP) | **Evaluated**: 5 × 5-fold CV on 70,000 records, in `results/runs/ledger.csv` |
| DCQF ablation (synthetic data) | **Evaluated**: `results/runs/dcqf_null_test.json`, written up in `docs/DCQF_FINDINGS.md` |
| FastAPI inference backend (`backend/`) | Implemented. Serves `backend/artifacts/inference_bundle.joblib` (hybrid DCQF + gradient boosting). This bundle is **not yet evaluated** |
| Hybrid model, VQC, quantum kernel, Control-C, noise benchmarks | Configured, **not yet evaluated** (no ledger rows) |
| Web frontend (`frontend/`) | Implemented: product-style React + TypeScript + Vite app |
| Streamlit prototype (`app/`) | Legacy. References a schema that no longer exists (see [Known issues](#known-issues)) |

---

## Architecture

```
                ┌────────────────────────── frontend/ (React, Vite) ───────────────────────────┐
                │  Landing · Dashboard · Assessment · History · Explainability · Benchmarks ·  │
                │  Quantum Hardware · Methodology · About                                      │
                └───────────────┬──────────────────────────────────────────────────────────────┘
                                │  /api/*  (Vite dev/preview proxy → :8000)
                ┌───────────────▼────────────── backend/ (FastAPI) ────────────────────────────┐
                │  GET /api/health   POST /api/predict   GET /api/benchmark                    │
                │  loads one frozen inference bundle (backend/artifacts/*.joblib|*.pkl)        │
                └───────────────┬──────────────────────────────────────────────────────────────┘
                                │  imports
                ┌───────────────▼────────────── src/qheart (research package) ─────────────────┐
                │  schema · features · preprocess · quantum (DCQF, VQC, kernel, noise) ·       │
                │  models · eval (harness, metrics, stats) · explain · report                  │
                └───────────────┬──────────────────────────────────────────────────────────────┘
                                │  append-only
                        results/runs/ledger.csv   ·   results/runs/dcqf_null_test.json
```

### Inference pipeline (what `/api/predict` runs)

```
11 raw clinical inputs
  → clinical features (age in years, BMI, pulse pressure, mean arterial pressure)
  → fold-safe feature selection (MI-mRMR, k = 8)
  → imputation + angle scaling to [0, π]                          8 classical features
  → DCQF circuit: 8 qubits, MI-weighted ZZ couplings,
    1 counterdiabatic Trotter step, Z-string readout (orders 1–3)  24 quantum features
  → concatenate                                                    32 hybrid features
  → fitted StandardScaler
  → GradientBoostingClassifier
  → P(cardio = 1): threshold 0.5 → prediction + risk label
```

The backend reuses the research objects (`ClinicalRepresentation`, `FoldSafeFeatureSelector`, `DCQFExtractor`) instead of reimplementing them, and it never trains during a request.

---

## Repository layout

```
SwasthyaQuant/
├── frontend/                 Web UI (React 19, TypeScript, Vite, Tailwind CSS v4, Recharts)
├── backend/                  FastAPI inference service
│   ├── app/api/routes/       health.py · predict.py · benchmark.py
│   ├── app/schemas/          patient.py (request) · prediction.py (response)
│   ├── app/services/         model, preprocessing, prediction, benchmark services
│   └── tests/                API contract tests
├── src/qheart/               Research package
│   ├── data/                 loaders, audit, splits
│   ├── features/             clinical feature engineering, fold-safe selection, PCA
│   ├── preprocess/           ClinicalRepresentation, fixed angle scaler
│   ├── quantum/              dcqf.py, vqc.py, kernel.py, extractor.py, ansatz.py, noise.py, statevector.py
│   ├── models/               registry, classical models, Control-C, DCQF controls
│   ├── eval/                 harness, metrics (pure numpy), stats, subgroups
│   ├── explain/              SHAP (classical), parameter-shift saliency (quantum)
│   └── report/               tables, figures
├── configs/                  base.yaml + data / features / model configs
├── data/                     cardio_train.csv (raw), processed CSVs, DATA_CARD.md
├── results/runs/             ledger.csv, dcqf_null_test.json
├── scripts/                  DCQF experiments, data fetch, split validation
├── docs/                     ARCHITECTURE, DECISIONS (ADRs), RESULTS_PROTOCOL, DCQF_FINDINGS
├── tests/                    research-package unit tests
└── app/                      legacy Streamlit prototype
```

---

## Quick start

Prerequisites: **Node.js 20+** and **Python 3.10–3.12**.

### 1. Backend (FastAPI)

**macOS / Linux**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=../src uvicorn app.main:app --reload --port 8000
```

**Windows (PowerShell)**

```powershell
cd backend
# One-time setup. Needs Python 3.10–3.12 (see the note below). With uv installed:
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
# ...or, if Python 3.12 is installed:  py -3.12 -m venv .venv; .venv\Scripts\python -m pip install -r requirements.txt

# Every time:
$env:PYTHONPATH = "..\src"; .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

If pip prints `Preparing metadata (pyproject.toml) ... error` for scikit-learn, the venv was created with Python 3.13+. Delete `backend\.venv` and recreate it with one of the commands above.

Swagger UI: <http://localhost:8000/docs>.

> **Use the pinned versions, especially `scikit-learn==1.5.2`.** The committed model bundle was saved with scikit-learn 1.5.2, and newer versions cannot unpickle it. Python 3.13+ has no wheels for these pins, so use **Python 3.10–3.12**. If the bundle fails to load, the backend silently trains a fallback model at startup (first 10,000 rows) and **overwrites `backend/artifacts/inference_bundle.joblib`**. If `git status` shows that file modified after starting the backend, your environment is wrong: run `git restore backend/artifacts/inference_bundle.joblib` and reinstall.
>
> No Python 3.12? With [uv](https://docs.astral.sh/uv/): `uv venv --python 3.12 .venv` then `uv pip install -r requirements.txt`.

When the model loads, `/api/health` returns `{"status": "ok", "model_loaded": true}`. Without a model, `/api/health` reports `"degraded"` and `/api/predict` returns **503**.

### 2. Frontend (React)

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 (proxies /api to http://127.0.0.1:8000)
```

Other scripts:

| Command | Purpose |
|---|---|
| `npm run build` | Type-check and produce a production build in `frontend/dist/` |
| `npm run preview` | Serve the production build on :4173 (also proxies `/api`) |
| `npm run typecheck` | TypeScript only |
| `npm run lint` | oxlint |
| `npm run test:api` | Frontend ↔ backend integration check (see [Testing](#testing)) |

Frontend environment variables (copy `frontend/.env.example` to `frontend/.env.local`):

| Variable | Default | Meaning |
|---|---|---|
| `VITE_API_PROXY_TARGET` | `http://127.0.0.1:8000` | Where the dev/preview server forwards `/api` |
| `VITE_API_BASE_URL` | *(empty)* | Absolute API origin for deployments. Leave empty for same-origin `/api` |
| `VITE_DEMO_MODE` | `false` | When `true` **and** the model is unavailable, the assessment screen shows a clearly watermarked *illustrative placeholder* instead of an error. Never use it to present real results |

The backend has no CORS middleware, so the frontend calls it through the Vite proxy. A production deployment should serve both behind one origin (for example, a reverse proxy routing `/api` to Uvicorn).

### 3. Research pipeline

```bash
make setup            # core deps + editable install of qheart
make setup-quantum    # PennyLane, Qiskit/Aer, PyTorch, SHAP, Streamlit
make audit            # dataset audit: run before any model
make baselines        # five classical baselines, 5 × 5-fold CV → results/runs/ledger.csv
make control          # parameter-matched Control-C
make kernel           # quantum kernel → SVC (check `make budget` first)
make vqc              # variational quantum classifier
make compare          # aggregate the ledger; refuses an incomplete table
make test             # pytest (quantum/data tests skip when deps are absent)
```

DCQF null test: `PYTHONPATH=src python scripts/dcqf_null_test.py --seeds 4 --n 400` (~96 s).

Backend tests: `PYTHONPATH=src pytest backend/tests/test_api.py -q`.

---

## API reference

All endpoints are defined in `backend/app/api/routes/`.

### `GET /api/health`

```json
{ "status": "ok" | "degraded", "model_loaded": true | false }
```

### `POST /api/predict`

Request body (`backend/app/schemas/patient.py`, extra fields are rejected):

| Field | Type | Range / values | Notes |
|---|---|---|---|
| `age` | int | 18–120 | years (converted to days internally) |
| `height` | int | 100–250 | cm |
| `weight` | float | 20–300 | kg |
| `ap_hi` | int | 60–250 | systolic BP, mmHg, must be ≥ `ap_lo` |
| `ap_lo` | int | 30–150 | diastolic BP, mmHg |
| `gender` | int | 1, 2 | dataset coding: 1 = female, 2 = male |
| `cholesterol` | int | 1, 2, 3 | normal / above normal / well above normal |
| `gluc` | int | 1, 2, 3 | normal / above normal / well above normal |
| `smoke`, `alco`, `active` | int | 0, 1 | smoker, alcohol intake, physically active |

Response `200`:

```json
{
  "prediction": 1,
  "risk_probability": 0.73,
  "risk_percentage": 73,
  "risk_label": "Higher estimated risk",
  "disclaimer": "Model-estimated risk only. This is not a medical diagnosis or clinical advice."
}
```

Errors: `422` returns a Pydantic validation list (`detail[].loc`, `detail[].msg`). `503` means no model artifact is loaded (`detail` explains why).

### `GET /api/benchmark`

```json
{ "status": "available", "source": "results/runs/ledger.csv",
  "models": { "<model>": { "count": 25, "latest": { "...": "latest ledger row as strings" } } } }
```

See [Known issues](#known-issues): this endpoint currently returns an empty `models` object.

---

## Web application

The UI is a product-style workspace: a navigation sidebar (collapsible on tablet, a drawer on mobile), a top bar with live service status, and five sections.

| Route | Screen | Data source |
|---|---|---|
| `/` | **Risk Assessment** (home). Patient-profile panel on the left (demographics, body measurements, blood pressure, lab markers, lifestyle switches) with the API's validation ranges. On the right: live health metrics (BMI, blood pressure category, pulse pressure, MAP) and the result card (gauge, verdict, probability bar with decision threshold, how it was computed) | `POST /api/predict` |
| `/history` | **History**: past assessments saved on this device | `localStorage` |
| `/insights` | **Insights**: latest assessment, feature contributions (ready for an attribution endpoint), how a prediction is made, what the model looks at | latest assessment, feature rationale |
| `/performance` | **Model Performance**: metric switcher, model comparison chart, full mean ± sd table, confusion matrix, ROC placeholder | `GET /api/benchmark` + ledger |
| `/quantum` | **Quantum Engine**: 8-qubit DCQF circuit diagram, execution modes, noise benchmark placeholder, quantum feature validation (ablation) | configs, `dcqf_null_test.json` |

**Data integrity rules the UI follows**

- No metric is hard-coded. Classical metrics are aggregated in the browser from `results/runs/ledger.csv`, which is bundled at build time: only 5 × 5-fold runs, mean ± sd, never a single split.
- Anything without data (hybrid, VQC, kernel and Control-C metrics, ROC points, noise benchmarks, SHAP) is shown as **"Awaiting benchmark"** or **"Not available"**, never as placeholder numbers.
- A prediction appears only if `/api/predict` returned it. Demo output is opt-in, isolated in `src/mocks/mockData.ts`, and watermarked on every screen it touches.
- Derived clinical features (BMI, pulse pressure, MAP, BP group) are recomputed in the browser for display with the formulas from `src/qheart/features/clinical.py`. The backend computes its own.

**Frontend structure**

```
frontend/src/
├── components/
│   ├── layout/       AppLayout, Sidebar (collapsible; drawer on mobile), ServiceStatusPill, Brand
│   ├── ui/           Card, StatCard, StatusBadge, Loading/Empty/Error states, Disclaimer, Callout, InfoTip, Segmented
│   ├── clinical/     form fields (number, segmented, switch), PredictionResult
│   ├── charts/       RiskGauge, ModelComparisonChart, BenchmarkTable, ConfusionMatrix, FeatureImportanceChart, DcqfAblationChart
│   └── quantum/      ModelPipeline, QuantumCircuitVisualization
├── pages/            one file per route
├── services/         predictionService, benchmarkService, healthService
├── lib/              api client, ledger parser/aggregator, clinical formulas, formatting, history
├── data/research.ts  repo-sourced constants, each citing its source file
├── mocks/            demo-only placeholder (off by default)
└── types/api.ts      TypeScript mirror of the backend schemas
```

---

## Research pipeline

- **Dataset**: Cardiovascular Disease dataset, 70,000 records, 11 features, binary `cardio` target (`data/raw/cardio_train.csv`, `configs/base.yaml`). Both classes sit near 50% prevalence.
- **Fold safety**: imputation, scaling and feature selection are fitted inside each training fold. The quantum angle scaler uses fixed physiological bounds, so it has no fitted state.
- **Evaluation protocol** (`docs/RESULTS_PROTOCOL.md`, written before any model was fitted):
  - 5 repeats × stratified 5-fold, reported as mean ± sd.
  - Decision threshold chosen on training folds by Youden's J.
  - Headline metrics: sensitivity and specificity. Primary discriminative metric: PR-AUC.
  - McNemar and Nadeau–Bengio corrected t-tests, with Holm–Bonferroni correction.
- **Quantum arms**:
  - **DCQF**: 8 qubits, 24 features, 0 trainable quantum parameters.
  - **VQC**: 4 qubits, depth 6, 96 trainable angles.
  - **ZZ quantum kernel + SVC**.
  - **Fixed random quantum extractor + logistic head**, compared against the parameter-matched **Control-C**.
- **Noise and hardware** (`src/qheart/quantum/noise.py`):
  - Exact statevector.
  - Finite-shot grid from 32 to 8,192 shots.
  - Qiskit Aer depolarising noise (1q 0.001, 2q 0.01, readout 0.02).
  - IBM Quantum Runtime sampler.
- **Design decisions**: `docs/DECISIONS.md` (ADR-001 … ADR-010).

---

## Results so far

### Classical baselines: 5 × 5-fold CV, `cardio_70000`, 8 compact features

Values are rounded from `results/runs/ledger.csv` (mean over 25 folds). The web app's Benchmarks page shows the full table with standard deviations.

| Model | Sensitivity | Specificity | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.698 | 0.748 | 0.769 | 0.784 | 0.716 |
| SVM (RBF) | 0.693 | 0.758 | 0.777 | 0.791 | 0.716 |
| Random Forest | 0.729 | 0.640 | 0.734 | 0.749 | 0.698 |
| XGBoost | 0.703 | 0.751 | 0.779 | 0.793 | 0.720 |
| MLP (32-16) | 0.697 | 0.753 | 0.776 | 0.791 | 0.717 |

**No hybrid or quantum model has been evaluated on this dataset yet.**

### DCQF ablation: synthetic data (n = 400, 4 seeds × 5 folds)

DCQF did not beat its own scrambled null in any label regime. The classical chain-product control matched it (pairwise regime) or beat it (linear and mixed regimes). Its apparent gain over raw features comes from expanding 8 columns to 24, not from the mutual-information encoding. Full analysis, including a selection-leakage issue in the source paper's headline result, is in `docs/DCQF_FINDINGS.md`.

---

## Testing

### 1. Automated integration check (recommended)

With the backend (port 8000) and the frontend dev server (port 5173) both running:

```bash
cd frontend
npm run test:api
```

This sends real requests **through the Vite dev server's `/api` proxy**, the same path the browser uses. It checks every endpoint against the response shapes the UI expects:

| Check | Expected |
|---|---|
| Frontend served at `/` | `index.html` loads |
| `GET /api/health` | `200`, `{status, model_loaded}` |
| `POST /api/predict` (valid patient) | `200`; `prediction` is 0/1, `risk_probability` in [0, 1], `risk_percentage = round(p × 100)`, prediction consistent with the 0.5 threshold. If no model is loaded, `503` instead |
| `POST /api/predict` (age 10, systolic < diastolic) | `422`, naming the bad fields |
| `POST /api/predict` (unknown field / missing field) | `422` |
| `GET /api/benchmark` | `200`, `{status, models}` |

It prints `PASS`/`FAIL` per check and exits with code 1 on any failure. To test FastAPI directly, bypassing the proxy: `API_URL=http://127.0.0.1:8000 npm run test:api` (PowerShell: `$env:API_URL="http://127.0.0.1:8000"; npm run test:api`).

### 2. Manual test in the browser

Open <http://localhost:5173>, then:

1. **Connection**: the pill in the top-right reads **Model ready** (green). "Model not loaded" (amber) means the API is up without a model. "API offline" (red) means the backend isn't running.
2. **Higher-risk patient**: keep the sample patient (58 y, female, 170 cm, 78 kg, BP 145/90) and click **Analyze Risk**. Expect about **82%, Higher estimated risk**.
3. **Lower-risk patient**: set Male, 35 y, 178 cm, 70 kg, BP 115/75. Expect about **15%, Lower estimated risk**.
4. **Validation**: set Age = 10, or Systolic 80 with Diastolic 95. The fields are highlighted with messages, focus jumps to the first bad field, and **no request is sent** (DevTools → Network).
5. **History**: open History. Each completed assessment is listed and can be reopened.
6. **Backend down**: stop Uvicorn. The pill turns to **API offline**, and Analyze Risk shows "Unable to reach the risk engine" instead of a result.

Exact percentages depend on the committed bundle and will change when the model is retrained.

### 3. Backend tests

```bash
PYTHONPATH=src pytest backend/tests/test_api.py -q
```

## Known issues

These were found during the frontend integration. None were changed, because backend and ML code were out of scope for the UI work.

1. **The served model is not evaluated.** The committed bundle was trained on the first 10,000 rows (80 trees) with no held-out evaluation, so the UI shows its validated accuracy as "Pending". Its predictions are for demonstrating the pipeline only. The backend also retrains and overwrites the bundle whenever loading fails (see [Quick start](#quick-start)).
2. **`GET /api/benchmark` returns `"models": {}`.** `backend/app/api/routes/benchmark.py` splits CSV lines on raw commas. Every ledger row has a quoted `config_json` containing commas, so all 137 rows fail its length check and are skipped. Using Python's `csv` module would fix it. Until then, the frontend aggregates the same file with a quote-aware parser and labels the source.
3. **The benchmark endpoint returns only the latest single fold per model.** The results protocol forbids reporting a single split, so the frontend always aggregates all folds itself.
4. **Legacy Streamlit app is broken.** `app/streamlit_app.py` reads `S.CATEGORIES["sex"]`, `"chest_pain"` and others from the earlier heart-failure schema. The current schema is the 70k cardiovascular one.
5. **Some docs describe the earlier dataset.** `data/DATA_CARD.md`, `docs/ARCHITECTURE.md` and parts of `docs/DECISIONS.md` describe the 918-row UCI/Kaggle heart-failure dataset. Code, configs, backend and ledger use `cardio_70000`.
6. **No CORS middleware on the backend.** The frontend relies on its dev/preview proxy.

---

## Demo flow (2–3 minutes)

1. **Risk Assessment** (`/`): the sample patient is preloaded. Point out the patient-profile panel, the live health metrics (BMI and blood-pressure category update as you type), and the ranges the API enforces. Click **Analyze Risk**.
   - Show the gauge, the verdict, the probability bar against the 50% decision threshold, and **How this was computed**.
   - Change the patient to a 35-year-old with BP 115/75 and re-run to show a lower-risk result.
2. **Model Performance** (`/performance`):
   - Switch between sensitivity and PR-AUC.
   - Note XGBoost's lead on PR-AUC/ROC-AUC and Random Forest's on sensitivity.
   - Open the confusion matrix.
   - Point to the hatched rows: quantum and hybrid models are awaiting evaluation.
3. **Quantum Engine** (`/quantum`): walk the 8-qubit DCQF circuit, then the execution modes from ideal simulation to IBM hardware. Close on **Quantum feature validation**: the quantum component was tested against a scrambled null and a classical equivalent, and the product reports that it did not beat them.
4. **Insights** (`/insights`): how a prediction is made, and the clinical factors the model considers.

---

## Scope and limitations

- **No quantum-advantage claim.** XGBoost is expected to win the accuracy column.
- **No clinical-utility claim.** This is not a medical device.
- **No generalisation claim.** Nothing supports generalisation to new hospitals or populations.
- **No hardware claim.** A simulator result is never labelled as hardware.
- **Class imbalance.** The dataset is imbalanced by gender (45,530 female vs 24,470 male records). Subgroup reporting is planned (`src/qheart/eval/subgroups.py`).

---

## References

- Simen, Flores-Garrigós, De Oliveira, Alvarado Barrios, Gomez Cadavid, Dalal, Solano, Hegade & Zhang. *Digitized counterdiabatic quantum feature extraction.* Scientific Reports (2026), accepted article in press. [doi:10.1038/s41598-026-67564-0](https://doi.org/10.1038/s41598-026-67564-0). Implemented as `DCQFExtractor`. Quotations should be re-checked against the final version.

---

## Technology

- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS v4, Recharts, React Router, lucide-react
- **API**: Python, FastAPI, Pydantic, Uvicorn
- **ML**: scikit-learn, XGBoost, NumPy, pandas, SciPy
- **Quantum**: PennyLane, Qiskit, Qiskit Aer, Qiskit Machine Learning, IBM Quantum Runtime, PyTorch
- **Explainability**: SHAP, parameter-shift saliency
