// Mirrors backend/app/schemas/*.py and backend/app/api/routes/*.py exactly.
// Do not add fields here that the backend does not send.

/** POST /api/predict request body — backend/app/schemas/patient.py (extra="forbid"). */
export interface PatientInput {
  /** Years, 18–120. The backend converts to days for the research pipeline. */
  age: number
  /** cm, 100–250 */
  height: number
  /** kg, 20–300 */
  weight: number
  /** Systolic BP, mmHg, 60–250, must be >= ap_lo */
  ap_hi: number
  /** Diastolic BP, mmHg, 30–150 */
  ap_lo: number
  smoke: 0 | 1
  alco: 0 | 1
  active: 0 | 1
  /** Dataset coding: 1 = female, 2 = male */
  gender: 1 | 2
  /** 1 normal, 2 above normal, 3 well above normal */
  cholesterol: 1 | 2 | 3
  /** 1 normal, 2 above normal, 3 well above normal */
  gluc: 1 | 2 | 3
}

/** POST /api/predict 200 response — backend/app/schemas/prediction.py */
export interface PredictionResponse {
  prediction: 0 | 1
  risk_probability: number
  risk_percentage: number
  risk_label: string
  disclaimer: string
}

/** GET /api/health */
export interface HealthResponse {
  status: 'ok' | 'degraded' | string
  model_loaded: boolean
}

/** One ledger row as returned by the API: every CSV column as a string. */
export type LedgerRowStrings = Record<string, string>

/** GET /api/benchmark */
export type BenchmarkResponse =
  | {
      status: 'available'
      source?: string
      models: Record<string, { count: number; latest: LedgerRowStrings }>
    }
  | { status: 'unavailable'; message: string; models: Record<string, never> }

/** FastAPI / Pydantic 422 error item. */
export interface ValidationIssue {
  type: string
  loc: (string | number)[]
  msg: string
}
