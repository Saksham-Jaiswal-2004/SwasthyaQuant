import { createContext, useContext } from 'react'
import type { PredictionRecord } from '../services/predictionService'
import type { HealthResponse } from '../types/api'

export type ServiceStatus =
  | { state: 'checking' }
  | { state: 'online'; health: HealthResponse }
  | { state: 'offline' }

export interface AppStateValue {
  service: ServiceStatus
  refreshHealth: () => void
  history: PredictionRecord[]
  addRecord: (r: PredictionRecord) => void
  removeRecord: (id: string) => void
  clearHistory: () => void
  latest?: PredictionRecord
}

export const AppStateContext = createContext<AppStateValue | null>(null)

export function useAppState() {
  const v = useContext(AppStateContext)
  if (!v) throw new Error('useAppState must be used inside AppStateProvider')
  return v
}

export const modelLoaded = (s: ServiceStatus) => s.state === 'online' && s.health.model_loaded
