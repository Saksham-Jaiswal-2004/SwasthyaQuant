import type { PredictionRecord } from '../services/predictionService'

// Prediction history is kept only in this browser (the backend has no history endpoint).
const KEY = 'swasthya-quant.history.v1'
const MAX = 50

export function loadHistory(): PredictionRecord[] {
  try {
    const raw = localStorage.getItem(KEY)
    const parsed = raw ? (JSON.parse(raw) as PredictionRecord[]) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export function saveHistory(items: PredictionRecord[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(items.slice(0, MAX)))
  } catch {
    /* storage unavailable: history stays in memory for this session */
  }
}
