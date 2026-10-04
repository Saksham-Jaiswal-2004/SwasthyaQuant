# QHeart FastAPI backend

This backend is a thin production interface over the existing qheart research codebase. It is intentionally not a replacement model or simplified feature pipeline.

## Architecture

The backend uses the repository's real research pipeline as the source of truth:

Raw 11 features
    ↓
ClinicalRepresentation
    ↓
8 clinical features
    ↓
DCQFExtractor
    ↓
24 quantum features
    ↓
32 hybrid features
    ↓
StandardScaler
    ↓
trained GradientBoostingClassifier
    ↓
positive-class probability

## Relationship to the research code

This project remains in the `src/qheart` package. The backend imports and reuses:

- `qheart.schema`
- `qheart.features.clinical.add_clinical_features`
- `qheart.features.select.FoldSafeFeatureSelector`
- `qheart.preprocess.pipeline.ClinicalRepresentation`
- `qheart.quantum.dcqf.DCQFExtractor`

The backend does not recreate these in a simplified form. It validates the request using the canonical schema and then applies the same logic from the research pipeline to one patient at a time.

## Model artifact requirement

The repository includes a frozen inference bundle at `backend/artifacts/inference_bundle.joblib`. The research benchmark tables are kept separately from the served artifact; the CV metrics shown in the UI are not recomputed at request time.

The expected bundle is a Python pickle or joblib dictionary with at least:

```python
{
    "clinical_representation": fitted_ClinicalRepresentation,
    "selector": fitted_feature_selector,
    "dcqf": fitted_DCQFExtractor,
    "scaler": fitted_StandardScaler,
    "model": fitted_GradientBoostingClassifier,
}
```

The backend loads the frozen artifact when available. A development-only fallback can rebuild a small prototype model from the bundled dataset if the artifact is unavailable; that fallback is not a benchmark result and must not be described as the validated 5×5 CV model.

## API endpoints

### GET /api/health

Returns model load status.

### POST /api/predict

Request body:

```json
{
  "age": 58,
  "height": 170,
  "weight": 78,
  "ap_hi": 145,
  "ap_lo": 90,
  "smoke": 0,
  "alco": 0,
  "active": 1,
  "gender": 1,
  "cholesterol": 1,
  "gluc": 1
}
```

Response:

```json
{
  "prediction": 1,
  "risk_probability": 0.73,
  "risk_percentage": 73,
  "risk_label": "Higher estimated risk",
  "disclaimer": "Model-estimated risk only. This is not a medical diagnosis or clinical advice."
}
```

### GET /api/benchmark

Returns the existing research benchmark ledger information if available.

## Run locally

From the backend directory:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=../src uvicorn app.main:app --reload
```

Swagger docs are available at:

- http://localhost:8000/docs

## Run tests

From the repository root:

```bash
PYTHONPATH=src pytest backend/tests/test_api.py -q
```

## Benchmark vs. deployed inference — do not conflate them

The website's Stage B.2/B.3/B.4 tables are offline research evaluations. They compare representations under the documented group-aware protocols and are not recalculated by the API. The deployed API uses the frozen `backend/artifacts/inference_bundle.joblib` for patient-level inference. Therefore: **benchmark metrics shown in the UI are evidence about the research experiments, not a guarantee that every API prediction reproduces a particular CV fold.**

The current research results do **not** establish quantum advantage: the 5×5 benchmark's classical baseline has ROC-AUC 0.7891 versus 0.7878 for the all-feature hybrid arm. This is intentionally reported as a limitation rather than hidden by the prototype.

## Why the backend does not retrain on requests

This project keeps research evaluation and production inference separate. The evaluation code produces cross-validation metrics and result logs, but those are not frozen inference artifacts. The backend requires a single trained model bundle and loads it once into memory. It never fits a new scaler or model during requests.

## Medical / non-diagnostic disclaimer

The model is not a medical device and does not diagnose disease. It outputs an estimated risk score from a research-trained model only.
