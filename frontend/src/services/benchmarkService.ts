import { LEDGER_SNAPSHOT } from '../data/research'
import { request } from '../lib/api'
import { parseLedger, summarise, type ModelSummary } from '../lib/ledger'
import type { BenchmarkResponse } from '../types/api'

export interface LiveBenchmarkStatus {
  reachable: boolean
  status?: string
  modelCount: number
  /** Ledger rows the API reported per model (its payload holds only the latest fold). */
  counts: Record<string, number>
  message?: string
}

export interface BenchmarkData {
  summaries: ModelSummary[]
  /** Where `summaries` came from. */
  origin: 'snapshot'
  sourcePath: string
  totalRows: number
  live: LiveBenchmarkStatus
}

/**
 * GET /api/benchmark returns only each model's latest single-fold row, which the results
 * protocol forbids reporting. Aggregated mean ± sd therefore always comes from the ledger
 * file itself (bundled at build time). The live endpoint is still called and its status
 * surfaced, so the UI shows exactly what the backend currently serves.
 */
export async function getBenchmarks(): Promise<BenchmarkData> {
  const rows = parseLedger(LEDGER_SNAPSHOT.text)
  let live: LiveBenchmarkStatus
  try {
    const res = await request<BenchmarkResponse>('/api/benchmark')
    const counts = Object.fromEntries(Object.entries(res.models ?? {}).map(([k, v]) => [k, v.count]))
    live = {
      reachable: true,
      status: res.status,
      modelCount: Object.keys(counts).length,
      counts,
      message: res.status === 'unavailable' ? res.message : undefined,
    }
  } catch {
    live = { reachable: false, modelCount: 0, counts: {} }
  }
  return { summaries: summarise(rows), origin: 'snapshot', sourcePath: LEDGER_SNAPSHOT.path, totalRows: rows.length, live }
}
