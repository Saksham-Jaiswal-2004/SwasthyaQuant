import { BarChart3, Database, Grid2x2, Layers, LineChart, RefreshCw, Repeat, Table2 } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { BenchmarkTable } from '../components/charts/BenchmarkTable'
import { ConfusionMatrix } from '../components/charts/ConfusionMatrix'
import { ModelComparisonChart } from '../components/charts/ModelComparisonChart'
import { Button, Card, CardHeader, EmptyState, ErrorState, InfoTip, LoadingState, PageHeader, Segmented } from '../components/ui/primitives'
import { DATASET, modelName } from '../data/research'
import { useAsync } from '../hooks/useAsync'
import { fmtInt } from '../lib/format'
import { metricDef, type MetricKey } from '../lib/ledger'
import { getBenchmarks } from '../services/benchmarkService'

const CHART_METRICS: MetricKey[] = ['sensitivity', 'specificity', 'pr_auc', 'roc_auc', 'f1', 'ppv', 'accuracy', 'fit_seconds']

export function BenchmarksPage() {
  const bench = useAsync(getBenchmarks)
  const [metric, setMetric] = useState<MetricKey>('sensitivity')
  const [cmModel, setCmModel] = useState<string>('')

  const data = bench.data
  const summaries = data?.summaries ?? []
  const cm = summaries.find((s) => s.model === (cmModel || summaries[0]?.model))
  const def = metricDef(metric)

  return (
    <>
      <PageHeader
        title="Model Performance"
        description="How each model performs at detecting cardiovascular disease, measured on held-out patients."
        actions={<Button variant="secondary" size="sm" onClick={bench.reload} disabled={bench.status === 'loading'}><RefreshCw className="size-4" aria-hidden /> Refresh</Button>}
      />

      {bench.status === 'loading' && !data && <Card><LoadingState label="Loading performance results…" /></Card>}
      {bench.status === 'error' && !data && <Card><ErrorState title="Performance data could not be loaded" onRetry={bench.reload} /></Card>}

      {data && (summaries.length === 0 ? (
        <Card><EmptyState icon={<BarChart3 className="size-5" aria-hidden />} title="Benchmark data is not available yet" action={<Button size="sm" variant="secondary" onClick={bench.reload}>Refresh</Button>}>Benchmark results will appear here after evaluation.</EmptyState></Card>
      ) : (
        <div className="space-y-5">
          <div className="grid gap-3 sm:grid-cols-3">
            <Fact icon={<Database className="size-4" aria-hidden />} k="Evaluation data" v={`${fmtInt(DATASET.records)} patients`} />
            <Fact icon={<Repeat className="size-4" aria-hidden />} k="Validation" v="5 × 5-fold cross-validation" />
            <Fact icon={<Layers className="size-4" aria-hidden />} k="Models evaluated" v={`${summaries.length} of ${summaries.length + 4}`} />
          </div>

          <Card>
            <CardHeader
              title={<span className="inline-flex items-center gap-1.5">{def.label} by model <InfoTip text={def.help} /></span>}
              subtitle="Average across all validation folds. The best result is highlighted."
              icon={<BarChart3 className="size-4" aria-hidden />}
            />
            <div className="space-y-4 p-5">
              <div className="overflow-x-auto pb-1">
                <Segmented label="Metric" size="sm" value={metric} onChange={setMetric} options={CHART_METRICS.map((k) => ({ value: k, label: metricDef(k).short }))} />
              </div>
              <ModelComparisonChart summaries={summaries} metric={metric} />
              <p className="sr-only">{summaries.map((s) => `${modelName(s.model)}: ${s.mean[metric]?.toFixed(3)}`).join('; ')}</p>
            </div>
          </Card>

          <Card>
            <CardHeader title="All metrics" subtitle="Mean ± standard deviation across folds. Quantum and hybrid models appear once their evaluation completes." icon={<Table2 className="size-4" aria-hidden />} />
            <BenchmarkTable summaries={summaries} highlight={metric} />
          </Card>

          <div className="grid gap-5 lg:grid-cols-2">
            <Card>
              <CardHeader title="Confusion matrix" subtitle="Correct and incorrect predictions on held-out patients" icon={<Grid2x2 className="size-4" aria-hidden />}
                action={
                  <label className="text-sm">
                    <span className="sr-only">Model</span>
                    <select value={cm?.model} onChange={(e) => setCmModel(e.target.value)} className="h-8 rounded-md border border-line bg-surface px-2 text-sm">
                      {summaries.map((s) => <option key={s.model} value={s.model}>{modelName(s.model)}</option>)}
                    </select>
                  </label>
                } />
              <div className="p-5">
                {cm && <ConfusionMatrix {...cm.confusion} caption={`${modelName(cm.model)}. Each of the ${fmtInt(cm.n)} patients is scored once per validation round; counts are averaged over 5 rounds.`} />}
              </div>
            </Card>
            <Card>
              <CardHeader title="ROC curve" subtitle="Trade-off between detection and false alarms" icon={<LineChart className="size-4" aria-hidden />} />
              <EmptyState icon={<LineChart className="size-5" aria-hidden />} title="ROC curve not available yet">
                The area under the curve is reported in the table. The full curve will appear here once it is published.
              </EmptyState>
            </Card>
          </div>
        </div>
      ))}
    </>
  )
}

function Fact({ icon, k, v }: { icon: ReactNode; k: string; v: string }) {
  return (
    <Card as="div" className="flex items-center gap-3 p-4">
      <span className="grid size-9 place-items-center rounded-lg bg-navy-50 text-navy-700">{icon}</span>
      <div className="min-w-0">
        <p className="text-xs text-ink-3">{k}</p>
        <p className="truncate font-semibold text-navy-900">{v}</p>
      </div>
    </Card>
  )
}
