// DEMO-ONLY. Used exclusively when VITE_DEMO_MODE=true AND the real model is unavailable.
// These values are NOT produced by any model. Every consumer must render them with the
// "Demo data" watermark (PredictionRecord.source === 'demo').

import type { PatientInput, PredictionResponse } from '../types/api'

/**
 * Deterministic placeholder so the result layout can be previewed. It is a hash of the
 * inputs, deliberately not a clinical heuristic, so it cannot be mistaken for a model.
 */
export function mockPrediction(input: PatientInput): PredictionResponse {
  const s = JSON.stringify(input)
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  const p = 0.2 + ((h >>> 0) % 6000) / 10000 // 0.20–0.80
  const prediction = p >= 0.5 ? 1 : 0
  return {
    prediction,
    risk_probability: p,
    risk_percentage: Math.round(p * 100),
    risk_label: prediction ? 'Higher estimated risk' : 'Lower estimated risk',
    disclaimer: 'DEMO DATA: illustrative placeholder, not a model output.',
  }
}
