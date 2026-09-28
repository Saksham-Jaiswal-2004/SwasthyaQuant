import { ArrowRight, Atom, BarChart3, ClipboardPlus, Cpu, Database, Gauge, Lightbulb, Server } from 'lucide-react'
import { Link } from 'react-router-dom'
import { ModelPipeline, PipelineLegend } from '../components/quantum/ModelPipeline'
import { Card, CardHeader, PageHeader, StatCard, StatusBadge } from '../components/ui/primitives'
import { DATASET, DCQF_CONFIG, modelName, SERVED_MODEL } from '../data/research'
import { modelLoaded, useAppState } from '../hooks/appStateContext'
import { useAsync } from '../hooks/useAsync'
import { fmtDateTime, fmtInt, fmtScore } from '../lib/format'
import { getBenchmarks } from '../services/benchmarkService'

export function DashboardPage() {
  const { service, history } = useAppState()
  const bench = useAsync(getBenchmarks)
  const loaded = modelLoaded(service)
  const summaries = bench.data?.summaries ?? []
  const bestSens = [...summaries].sort((a, b) => (b.mean.sensitivity ?? 0) - (a.mean.sensitivity ?? 0))[0]
  const bestPr = [...summaries].sort((a, b) => (b.mean.pr_auc ?? 0) - (a.mean.pr_auc ?? 0))[0]

  const statusValue =
    service.state === 'checking' ? 'Checking…'
    : service.state === 'offline' ? 'API offline'
    : loaded ? 'Ready' : 'Model not loaded'
  const statusTone = service.state === 'checking' ? 'pending' : service.state === 'offline' ? 'risk' : loaded ? 'good' : 'warn'

  return (
    <>
      <PageHeader
        eyebrow="Overview"
        title="Clinical intelligence dashboard"
        description="System status, the served hybrid model, and the measured research baselines at a glance."
        actions={
          <Link to="/app/assess" className="inline-flex h-10 items-center gap-2 rounded-lg bg-navy-900 px-4 text-sm font-medium text-white hover:bg-navy-800">
            <ClipboardPlus className="size-4" aria-hidden /> New assessment
          </Link>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Prediction status" icon={<Server className="size-4" aria-hidden />} value={statusValue}
          badge={<StatusBadge tone={statusTone}>{service.state === 'online' ? `health: ${service.health.status}` : service.state}</StatusBadge>}
          hint="GET /api/health" />
        <StatCard label="Model" icon={<Cpu className="size-4" aria-hidden />} value={SERVED_MODEL.short} hint="DCQF features + Gradient Boosting" />
        <StatCard label="Current dataset" icon={<Database className="size-4" aria-hidden />} value="Cardiovascular 70k" hint={`${fmtInt(DATASET.records)} records · ${DATASET.rawFeatures} features`} />
        <StatCard label="Quantum circuit" icon={<Atom className="size-4" aria-hidden />} value={`${DCQF_CONFIG.qubits} qubits`} hint={`${DCQF_CONFIG.quantumFeatures} Z-string features · exact statevector`} />
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <StatCard label="Hybrid model accuracy" pending value="Not available" badge={<StatusBadge tone="pending">Awaiting benchmark</StatusBadge>} />
        <StatCard label="Hybrid sensitivity" pending value="Not available" badge={<StatusBadge tone="pending">Awaiting benchmark</StatusBadge>} />
        <StatCard label="Hybrid specificity" pending value="Not available" badge={<StatusBadge tone="pending">Awaiting benchmark</StatusBadge>} />
      </div>
      <p className="mt-2 text-xs text-ink-3">The research ledger does not yet contain evaluation rows for the hybrid model, so its metrics are not shown. Classical baselines below are measured values.</p>

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="Hybrid inference pipeline" subtitle="The exact chain executed by POST /api/predict" icon={<Gauge className="size-4" aria-hidden />} />
          <div className="space-y-4 p-5">
            <ModelPipeline compact />
            <PipelineLegend />
          </div>
        </Card>

        <Card>
          <CardHeader title="Measured classical baselines" subtitle="5 × 5-fold CV · cardio_70000" icon={<BarChart3 className="size-4" aria-hidden />} />
          <div className="p-5">
            {bench.status === 'loading' && !bench.data ? (
              <p className="text-sm text-ink-3">Loading benchmark ledger…</p>
            ) : summaries.length === 0 ? (
              <p className="text-sm text-ink-3">Benchmark results will appear here after evaluation.</p>
            ) : (
              <dl className="space-y-3 text-sm">
                <Row k="Best sensitivity" model={modelName(bestSens.model)} v={fmtScore(bestSens.mean.sensitivity)} />
                <Row k="Best PR-AUC" model={modelName(bestPr.model)} v={fmtScore(bestPr.mean.pr_auc)} />
                <Row k="Models evaluated" model="classical" v={String(summaries.length)} />
              </dl>
            )}
            <Link to="/app/benchmarks" className="mt-4 inline-flex items-center gap-1.5 text-sm font-medium text-teal-700 hover:text-teal-600">
              Open benchmarks <ArrowRight className="size-4" aria-hidden />
            </Link>
          </div>
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader title="Recent assessments" subtitle="Stored in this browser only"
            action={<Link to="/app/history" className="text-sm font-medium text-teal-700 hover:text-teal-600">View all</Link>} />
          {history.length === 0 ? (
            <p className="px-5 py-8 text-center text-sm text-ink-3">No assessments yet. Run one from Disease Assessment.</p>
          ) : (
            <ul className="divide-y divide-line">
              {history.slice(0, 4).map((r) => (
                <li key={r.id} className="flex items-center justify-between gap-3 px-5 py-3 text-sm">
                  <div className="min-w-0">
                    <p className="font-medium text-ink">{r.input.age} y · {r.input.ap_hi}/{r.input.ap_lo} mmHg</p>
                    <p className="text-xs text-ink-3">{fmtDateTime(r.createdAt)}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {r.source === 'demo' && <StatusBadge tone="warn">Demo</StatusBadge>}
                    <StatusBadge tone={r.result.prediction ? 'risk' : 'good'}>{r.result.risk_percentage}% · {r.result.risk_label}</StatusBadge>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>
        <Card>
          <CardHeader title="Explore" />
          <ul className="divide-y divide-line text-sm">
            {[
              { to: '/app/explain', icon: Lightbulb, t: 'Explainability', d: 'Feature rationale and attribution' },
              { to: '/app/quantum', icon: Atom, t: 'Quantum hardware', d: 'Circuit, noise model, DCQF ablation' },
              { to: '/app/methodology', icon: BarChart3, t: 'Methodology', d: 'Fold-safe evaluation protocol' },
            ].map((l) => (
              <li key={l.to}>
                <Link to={l.to} className="flex items-center gap-3 px-5 py-3 hover:bg-navy-50/60">
                  <l.icon className="size-4 text-teal-700" aria-hidden />
                  <span className="min-w-0 flex-1"><span className="block font-medium text-ink">{l.t}</span><span className="block text-xs text-ink-3">{l.d}</span></span>
                  <ArrowRight className="size-4 text-ink-3" aria-hidden />
                </Link>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </>
  )
}

function Row({ k, model, v }: { k: string; model: string; v: string }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt><span className="block text-ink-2">{k}</span><span className="block text-xs text-ink-3">{model}</span></dt>
      <dd className="text-lg font-semibold text-navy-900 tabular">{v}</dd>
    </div>
  )
}
