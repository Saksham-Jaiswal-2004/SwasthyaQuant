import { BarChart3, Grid2x2, LineChart, RefreshCw, Table2 } from 'lucide-react'
import { useState } from 'react'
import { BenchmarkTable } from '../components/charts/BenchmarkTable'
import { ConfusionMatrix } from '../components/charts/ConfusionMatrix'
import { ModelComparisonChart } from '../components/charts/ModelComparisonChart'
import { Button, Callout, Card, CardHeader, EmptyState, ErrorState, InfoTip, LoadingState, PageHeader, Segmented, SourceNote, StatusBadge } from '../components/ui/primitives'
import { modelName } from '../data/research'
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
        eyebrow="Benchmarks"
        title="Model comparison"
        description="Classical baselines against quantum and hybrid arms, evaluated under one fold-safe harness: 5 repeats × stratified 5-fold, decision threshold chosen on training folds only."
        actions={<Button variant="secondary" size="sm" onClick={bench.reload} disabled={bench.status === 'loading'}><RefreshCw className="size-4" aria-hidden /> Refresh</Button>}
      />

      {bench.status === 'loading' && !data && <Card><LoadingState label="Loading benchmark results…" /></Card>}
      {bench.status === 'error' && !data && <Card><ErrorState title="Benchmark data could not be loaded" onRetry={bench.reload} /></Card>}

      {data && (
        <div className="space-y-5">
          <LiveApiNote live={data.live} />

          {summaries.length === 0 ? (
            <Card><EmptyState icon={<BarChart3 className="size-5" aria-hidden />} title="Benchmark data is not available yet" action={<Button size="sm" variant="secondary" onClick={bench.reload}>Refresh</Button>}>Benchmark results will appear here after evaluation.</EmptyState></Card>
          ) : (
            <>
              <Card>
                <CardHeader
                  title={<span className="inline-flex items-center gap-1.5">{def.label} by model <InfoTip text={def.help} /></span>}
                  subtitle={`Mean ± sd across ${summaries[0].folds} folds. Best model highlighted in teal. Quantum and hybrid arms have no ledger rows yet.`}
                  icon={<BarChart3 className="size-4" aria-hidden />}
                />
                <div className="space-y-4 p-5">
                  <div className="overflow-x-auto pb-1">
                    <Segmented label="Metric" size="sm" value={metric} onChange={setMetric} options={CHART_METRICS.map((k) => ({ value: k, label: metricDef(k).short }))} />
                  </div>
                  <ModelComparisonChart summaries={summaries} metric={metric} />
                  <p className="sr-only">
                    {summaries.map((s) => `${modelName(s.model)}: ${s.mean[metric]?.toFixed(3)}`).join('; ')}
                  </p>
                  <SourceNote>{data.sourcePath} ({fmtInt(data.totalRows)} rows, bundled at build time)</SourceNote>
                </div>
              </Card>

              <Card>
                <CardHeader title="Full results table" subtitle="Sensitivity and specificity are the headline metrics; PR-AUC is the primary discriminative metric (docs/RESULTS_PROTOCOL.md)." icon={<Table2 className="size-4" aria-hidden />} />
                <BenchmarkTable summaries={summaries} highlight={metric} />
                <div className="border-t border-line px-5 py-3">
                  <p className="text-xs leading-relaxed text-ink-3">sd is a spread across folds, not a confidence interval: folds share training data, so it understates true uncertainty. Fit time is training wall-clock per fold, not inference latency.</p>
                </div>
              </Card>

              <div className="grid gap-5 lg:grid-cols-2">
                <Card>
                  <CardHeader title="Confusion matrix" subtitle="Out-of-fold predictions, averaged over the 5 repeats" icon={<Grid2x2 className="size-4" aria-hidden />}
                    action={
                      <label className="text-sm">
                        <span className="sr-only">Model</span>
                        <select value={cm?.model} onChange={(e) => setCmModel(e.target.value)} className="h-8 rounded-md border border-line bg-surface px-2 text-sm">
                          {summaries.map((s) => <option key={s.model} value={s.model}>{modelName(s.model)}</option>)}
                        </select>
                      </label>
                    } />
                  <div className="p-5">
                    {cm && <ConfusionMatrix {...cm.confusion} caption={`${modelName(cm.model)} · ${fmtInt(cm.n)} patients per repeat. Each repeat covers every patient once as a test case; counts are the mean across repeats. Colour shows the rate within each actual class.`} />}
                  </div>
                </Card>
                <Card>
                  <CardHeader title="ROC curve" subtitle="Receiver operating characteristic" icon={<LineChart className="size-4" aria-hidden />} />
                  <EmptyState icon={<LineChart className="size-5" aria-hidden />} title="ROC curve points are not available">
                    The ledger stores the summary ROC-AUC per fold (shown in the table) but not the curve coordinates, and the API does not expose them.
                    This panel will render once per-threshold points are published.
                  </EmptyState>
                </Card>
              </div>

              <Callout tone="warn" title="Parameter-matched control (Control-C) has not been run">
                The results protocol requires a quantum model to be reported beside a classical control of equal parameter count. Until the
                <span className="font-mono"> control_c</span> and quantum rows exist in the ledger, no quantum-versus-classical claim is made here.
              </Callout>
            </>
          )}
        </div>
      )}
    </>
  )
}

function LiveApiNote({ live }: { live: { reachable: boolean; status?: string; modelCount: number; message?: string } }) {
  if (!live.reachable) {
    return (
      <Callout tone="info" title="Showing the bundled ledger snapshot">
        <span className="inline-flex flex-wrap items-center gap-2">GET /api/benchmark is unreachable. <StatusBadge tone="risk">API offline</StatusBadge></span>
      </Callout>
    )
  }
  if (live.modelCount === 0) {
    return (
      <Callout tone="info" title="GET /api/benchmark returned no models, so aggregates are computed from the ledger file">
        The endpoint responded (status: {live.status}) with an empty model list. Its CSV reader splits on raw commas and skips every row whose quoted
        <span className="font-mono"> config_json</span> contains commas. The figures below are read from the same file with a quote-aware parser.
      </Callout>
    )
  }
  return (
    <Callout tone="good" title={`GET /api/benchmark is serving ${live.modelCount} models`}>
      The API returns only each model&rsquo;s latest single fold. Per the results protocol, the mean ± sd below is aggregated from every fold in the ledger.
    </Callout>
  )
}
