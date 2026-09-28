import { DEMO_MODE } from '../config'
import { ApiError, request } from '../lib/api'
import { mockPrediction } from '../mocks/mockData'
import type { PatientInput, PredictionResponse } from '../types/api'

export interface PredictionRecord {
  id: string
  createdAt: string
  input: PatientInput
  result: PredictionResponse
  /** 'api' = returned by POST /api/predict. 'demo' = illustrative placeholder, never real. */
  source: 'api' | 'demo'
  durationMs: number
}

export async function predict(input: PatientInput): Promise<PredictionRecord> {
  const started = performance.now()
  // randomUUID is unavailable on insecure origins (e.g. a LAN IP over http).
  const id = crypto.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  const base = { id, createdAt: new Date().toISOString(), input }
  try {
    const result = await request<PredictionResponse>('/api/predict', {
      method: 'POST',
      body: JSON.stringify(input),
    })
    return { ...base, result, source: 'api', durationMs: performance.now() - started }
  } catch (err) {
    const fallback = DEMO_MODE && err instanceof ApiError && (err.kind === 'unavailable' || err.kind === 'network')
    if (!fallback) throw err
    return { ...base, result: mockPrediction(input), source: 'demo', durationMs: performance.now() - started }
  }
}
