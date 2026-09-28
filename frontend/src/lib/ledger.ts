// Parsing and aggregation of results/runs/ledger.csv (one row per model x repeat x fold).
// Aggregation follows docs/RESULTS_PROTOCOL.md: 5 repeats x 5-fold CV, mean ± sd, and
// single-split runs are never reported.

export type MetricKey =
  | 'sensitivity' | 'specificity' | 'ppv' | 'npv' | 'f1' | 'roc_auc' | 'pr_auc'
  | 'balanced_accuracy' | 'accuracy' | 'mcc' | 'kappa' | 'brier' | 'ece' | 'fit_seconds'

export interface MetricDef {
  key: MetricKey
  label: string
  short: string
  help: string
  /** true when a smaller value is better */
  lowerIsBetter?: boolean
  /** render as a 0–1 score (3 dp) vs a raw number */
  unit?: 's'
}

export const METRICS: MetricDef[] = [
  { key: 'sensitivity', label: 'Sensitivity (recall)', short: 'Sensitivity', help: 'Share of true disease cases the model flags. The headline screening metric: a missed case is the costly error.' },
  { key: 'specificity', label: 'Specificity', short: 'Specificity', help: 'Share of disease-free patients the model correctly clears.' },
  { key: 'pr_auc', label: 'PR-AUC', short: 'PR-AUC', help: 'Area under the precision–recall curve. The project’s primary discriminative metric.' },
  { key: 'roc_auc', label: 'ROC-AUC', short: 'ROC-AUC', help: 'Area under the ROC curve (threshold-free ranking quality).' },
  { key: 'f1', label: 'F1 score', short: 'F1', help: 'Harmonic mean of precision and recall at the chosen threshold.' },
  { key: 'ppv', label: 'Precision (PPV)', short: 'Precision', help: 'Of patients flagged, the share who truly have disease.' },
  { key: 'npv', label: 'NPV', short: 'NPV', help: 'Of patients cleared, the share who are truly disease-free.' },
  { key: 'balanced_accuracy', label: 'Balanced accuracy', short: 'Bal. acc.', help: 'Mean of sensitivity and specificity.' },
  { key: 'accuracy', label: 'Accuracy', short: 'Accuracy', help: 'Share of all patients classified correctly. Reported last, per the results protocol.' },
  { key: 'mcc', label: 'MCC', short: 'MCC', help: 'Matthews correlation coefficient, robust to class balance.' },
  { key: 'brier', label: 'Brier score', short: 'Brier', help: 'Mean squared error of predicted probabilities. Lower is better.', lowerIsBetter: true },
  { key: 'ece', label: 'Calibration error (ECE)', short: 'ECE', help: 'Expected calibration error. Lower is better.', lowerIsBetter: true },
  { key: 'fit_seconds', label: 'Fit time per fold', short: 'Fit time', help: 'Wall-clock training time per fold, in seconds (training, not inference).', lowerIsBetter: true, unit: 's' },
]

export const metricDef = (k: MetricKey) => METRICS.find((m) => m.key === k)!

export interface LedgerRow {
  model: string
  config: Record<string, unknown>
  values: Partial<Record<MetricKey, number>>
  tn: number; fp: number; fn: number; tp: number
  n: number; nPos: number; rep: number; fold: number
}

export interface ModelSummary {
  model: string
  folds: number
  repeats: number
  featureSet?: string
  source?: string
  mean: Partial<Record<MetricKey, number>>
  sd: Partial<Record<MetricKey, number>>
  /** Out-of-fold confusion counts, averaged over repeats (each repeat covers every patient once). */
  confusion: { tn: number; fp: number; fn: number; tp: number }
  /** Patients per repeat. */
  n: number
}

/** RFC-4180-ish CSV parser: handles quoted fields with embedded commas and doubled quotes. */
export function parseCsv(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++ }
      else if (c === '"') quoted = false
      else field += c
    } else if (c === '"') quoted = true
    else if (c === ',') { row.push(field); field = '' }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(field); field = ''
      if (row.some((v) => v !== '')) rows.push(row)
      row = []
    } else field += c
  }
  row.push(field)
  if (row.some((v) => v !== '')) rows.push(row)
  return rows
}

const num = (v: string | undefined) => {
  if (v === undefined || v.trim() === '') return undefined
  const n = Number(v)
  return Number.isFinite(n) ? n : undefined
}

export function rowFromRecord(rec: Record<string, string>): LedgerRow | null {
  if (!rec.model) return null
  let config: Record<string, unknown> = {}
  try { config = rec.config_json ? JSON.parse(rec.config_json) : {} } catch { /* keep empty */ }
  const values: LedgerRow['values'] = {}
  for (const m of METRICS) {
    const v = num(rec[m.key])
    if (v !== undefined) values[m.key] = v
  }
  return {
    model: rec.model, config, values,
    tn: num(rec.tn) ?? 0, fp: num(rec.fp) ?? 0, fn: num(rec.fn) ?? 0, tp: num(rec.tp) ?? 0,
    n: num(rec.n) ?? 0, nPos: num(rec.n_pos) ?? 0, rep: num(rec.rep) ?? 0, fold: num(rec.fold) ?? 0,
  }
}

export function parseLedger(text: string): LedgerRow[] {
  const [header, ...body] = parseCsv(text)
  if (!header) return []
  return body
    .filter((r) => r.length === header.length)
    .map((r) => rowFromRecord(Object.fromEntries(header.map((h, i) => [h.trim(), r[i]]))))
    .filter((r): r is LedgerRow => r !== null)
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
const sd = (xs: number[]) => {
  if (xs.length < 2) return 0
  const m = mean(xs)
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1))
}

/**
 * Summarise the protocol-compliant runs (folds=5, repeats=5) per model. Models with only
 * smoke-test runs (e.g. 2-fold) are excluded rather than reported from a single split.
 */
export function summarise(rows: LedgerRow[], folds = 5, repeats = 5): ModelSummary[] {
  const eligible = rows.filter((r) => r.config.folds === folds && r.config.repeats === repeats)
  const byModel = new Map<string, LedgerRow[]>()
  for (const r of eligible) byModel.set(r.model, [...(byModel.get(r.model) ?? []), r])

  return [...byModel.entries()].map(([model, rs]) => {
    const m: ModelSummary['mean'] = {}
    const s: ModelSummary['sd'] = {}
    for (const def of METRICS) {
      const xs = rs.map((r) => r.values[def.key]).filter((v): v is number => v !== undefined)
      if (xs.length) { m[def.key] = mean(xs); s[def.key] = sd(xs) }
    }
    const reps = new Set(rs.map((r) => r.rep)).size || 1
    const sum = (k: 'tn' | 'fp' | 'fn' | 'tp') => rs.reduce((a, r) => a + r[k], 0) / reps
    return {
      model,
      folds: rs.length,
      repeats: reps,
      featureSet: rs[0].config.feature_set as string | undefined,
      source: rs[0].config.source as string | undefined,
      mean: m,
      sd: s,
      confusion: { tn: sum('tn'), fp: sum('fp'), fn: sum('fn'), tp: sum('tp') },
      n: rs.reduce((a, r) => a + r.n, 0) / reps,
    }
  })
}
