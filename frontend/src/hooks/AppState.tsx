import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'
import { loadHistory, saveHistory } from '../lib/history'
import { getHealth } from '../services/healthService'
import type { PredictionRecord } from '../services/predictionService'
import { AppStateContext, type AppStateValue, type ServiceStatus } from './appStateContext'

const POLL_MS = 30_000

export function AppStateProvider({ children }: { children: ReactNode }) {
  const [service, setService] = useState<ServiceStatus>({ state: 'checking' })
  const [history, setHistory] = useState<PredictionRecord[]>(loadHistory)

  const poll = useCallback(() => {
    getHealth().then(
      (health) => setService({ state: 'online', health }),
      () => setService({ state: 'offline' }),
    )
  }, [])

  const refreshHealth = useCallback(() => {
    setService({ state: 'checking' })
    poll()
  }, [poll])

  useEffect(() => {
    poll()
    const t = window.setInterval(poll, POLL_MS)
    return () => window.clearInterval(t)
  }, [poll])

  useEffect(() => saveHistory(history), [history])

  const value = useMemo<AppStateValue>(() => ({
    service,
    refreshHealth,
    history,
    latest: history[0],
    addRecord: (r) => setHistory((h) => [r, ...h]),
    removeRecord: (id) => setHistory((h) => h.filter((x) => x.id !== id)),
    clearHistory: () => setHistory([]),
  }), [service, refreshHealth, history])

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>
}
