import { MODEL_CATALOG, modelName } from '../../data/research'
import { cx, fmtScore } from '../../lib/format'
import { metricDef, type MetricKey, type ModelSummary } from '../../lib/ledger'
import { InfoTip, StatusBadge } from '../ui/primitives'

const DEFAULT_COLS: MetricKey[] = ['sensitivity', 'specificity', 'pr_auc', 'roc_auc', 'f1', 'ppv', 'balanced_accuracy', 'accuracy', 'fit_seconds']

/**
 * mean ± sd per model. Models defined in the repo but absent from the ledger are listed
 * as "Awaiting benchmark" rather than hidden or filled in.
 */
export function BenchmarkTable({ summaries, columns = DEFAULT_COLS, awaiting = ['hybrid_dcqf_gb', 'vqc_dense', 'qkernel_zz', 'control_c'], highlight }: { summaries: ModelSummary[]; columns?: MetricKey[]; awaiting?: string[]; highlight?: MetricKey }) {
  const best: Partial<Record<MetricKey, number>> = {}
  for (const c of columns) {
    const vals = summaries.map((s) => s.mean[c]).filter((v): v is number => v !== undefined)
    if (vals.length) best[c] = metricDef(c).lowerIsBetter ? Math.min(...vals) : Math.max(...vals)
  }
  const pending = awaiting.filter((id) => !summaries.some((s) => s.model === id))

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[860px] border-collapse text-sm">
        <caption className="sr-only">Model benchmark results, mean plus or minus standard deviation across cross-validation folds</caption>
        <thead>
          <tr className="border-b border-line text-left text-xs font-medium text-ink-3">
            <th scope="col" className="sticky left-0 bg-surface px-4 py-2.5 font-medium">Model</th>
            {columns.map((c) => {
              const d = metricDef(c)
              return (
                <th key={c} scope="col" className={cx('px-3 py-2.5 text-right font-medium', highlight === c && 'text-navy-900')}>
                  <span className="inline-flex items-center gap-1">{d.short}<InfoTip text={d.help} label={`About ${d.short}`} /></span>
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {summaries.map((s) => (
            <tr key={s.model} className="border-b border-line/70 hover:bg-navy-50/50">
              <th scope="row" className="sticky left-0 bg-surface px-4 py-2.5 text-left font-medium text-navy-900">
                <span className="block">{modelName(s.model)}</span>
                <span className="block text-xs font-normal text-ink-3">{s.folds} validation folds</span>
              </th>
              {columns.map((c) => {
                const v = s.mean[c]
                const isBest = v !== undefined && v === best[c]
                const unit = metricDef(c).unit
                return (
                  <td key={c} className={cx('px-3 py-2.5 text-right tabular', highlight === c && 'bg-navy-50/60')}>
                    <span className={cx(isBest ? 'font-semibold text-teal-700' : 'text-ink')}>{unit ? `${v?.toFixed(1) ?? '—'}${unit}` : fmtScore(v)}</span>
                    <span className="block text-[11px] text-ink-3">± {unit ? s.sd[c]?.toFixed(1) : fmtScore(s.sd[c])}</span>
                  </td>
                )
              })}
            </tr>
          ))}
          {pending.map((id) => {
            const info = MODEL_CATALOG.find((m) => m.id === id)
            return (
              <tr key={id} className="border-b border-line/70">
                <th scope="row" className="sticky left-0 bg-surface px-4 py-2.5 text-left font-medium text-ink-2">
                  <span className="block">{info?.name ?? id}</span>
                  <span className="block text-xs font-normal text-ink-3">{info?.family === 'control' ? 'Classical control' : info?.family === 'hybrid' ? 'Hybrid quantum-classical' : 'Quantum'}</span>
                </th>
                <td colSpan={columns.length} className="hatch px-3 py-2.5 text-center">
                  <StatusBadge tone="pending">Awaiting evaluation</StatusBadge>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
