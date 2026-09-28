import { ArrowDownRight, ArrowUpRight, Layers, Lightbulb, ListTree, UserRound } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { FeatureImportanceChart, type FeatureAttribution } from '../components/charts/FeatureImportanceChart'
import { ModelPipeline } from '../components/quantum/ModelPipeline'
import { SERVED_PIPELINE } from '../components/quantum/pipelineStages'
import { Card, CardHeader, EmptyState, PageHeader, StatusBadge } from '../components/ui/primitives'
import { FEATURE_RATIONALE } from '../data/research'
import { useAppState } from '../hooks/appStateContext'
import { deriveFeatures } from '../lib/clinical'
import { fmtDateTime, fmtNum } from '../lib/format'

/**
 * No attribution endpoint exists yet (SHAP tooling lives in src/qheart/explain). When one
 * is added, pass its payload here; the chart and factor lists render from it unchanged.
 */
const ATTRIBUTIONS: FeatureAttribution[] | null = null

export function ExplainabilityPage() {
  const { latest } = useAppState()
  const attributions = ATTRIBUTIONS
  const up = attributions?.filter((a) => a.value > 0).sort((a, b) => b.value - a.value) ?? []
  const down = attributions?.filter((a) => a.value < 0).sort((a, b) => a.value - b.value) ?? []
  const d = latest ? deriveFeatures(latest.input) : null

  return (
    <>
      <PageHeader title="Insights" description="What drives a patient's risk estimate, and how the model reads clinical data." />

      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader title="Latest assessment" subtitle={latest ? fmtDateTime(latest.createdAt) : 'No assessment yet'} icon={<UserRound className="size-4" aria-hidden />} />
          <div className="p-5">
            {!latest ? (
              <EmptyState title="No assessment to explain" action={<Link to="/" className="text-sm font-medium text-teal-700 hover:text-teal-600">Start a risk assessment →</Link>} />
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge tone={latest.result.prediction ? 'risk' : 'good'}>{latest.result.risk_label}</StatusBadge>
                  <span className="text-sm text-ink-2 tabular">{(latest.result.risk_probability * 100).toFixed(1)}% estimated risk</span>
                  {latest.source === 'demo' && <StatusBadge tone="warn">Demo data</StatusBadge>}
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  <FactorList title="Increasing risk" icon={<ArrowUpRight className="size-4 text-risk" aria-hidden />} items={up} />
                  <FactorList title="Reducing risk" icon={<ArrowDownRight className="size-4 text-navy-600" aria-hidden />} items={down} />
                </div>
                <dl className="grid grid-cols-2 gap-3 rounded-lg bg-slate-50 p-3 text-sm sm:grid-cols-4">
                  {[
                    ['BMI', `${fmtNum(d?.bmi)}`, d?.bmiClass],
                    ['Blood pressure', `${latest.input.ap_hi}/${latest.input.ap_lo}`, d?.bpGroup],
                    ['Pulse pressure', fmtNum(d?.pulsePressure, 0), 'mmHg'],
                    ['Mean arterial', fmtNum(d?.map, 0), 'mmHg'],
                  ].map(([k, v, s]) => (
                    <div key={k}>
                      <dt className="text-xs text-ink-3">{k}</dt>
                      <dd className="font-semibold text-navy-900 tabular">{v}</dd>
                      <dd className="text-xs text-ink-3">{s}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}
          </div>
        </Card>

        <Card>
          <CardHeader title="Feature contributions" subtitle="How much each factor moved the estimate" icon={<Lightbulb className="size-4" aria-hidden />} />
          {attributions ? (
            <div className="p-5"><FeatureImportanceChart data={attributions} /></div>
          ) : (
            <EmptyState title="Contributions not available yet">
              Per-patient feature contributions will appear here once they are enabled for the hybrid model.
            </EmptyState>
          )}
        </Card>
      </div>

      <Card className="mt-5">
        <CardHeader title="How a prediction is made" subtitle="Clinical data → quantum features → classifier" icon={<Layers className="size-4" aria-hidden />} />
        <div className="grid gap-6 p-5 lg:grid-cols-[300px_minmax(0,1fr)]">
          <ModelPipeline stages={SERVED_PIPELINE} orientation="vertical" />
          <div className="space-y-4 text-sm leading-relaxed text-ink-2">
            <Step n={1} t="Clinical preprocessing">Values outside a plausible range are treated as missing. BMI, pulse pressure and mean arterial pressure are computed from the raw inputs.</Step>
            <Step n={2} t="Feature selection">The eight most informative, least redundant clinical features are kept.</Step>
            <Step n={3} t="Quantum feature encoding">Each selected feature sets a rotation on its own qubit. Stronger statistical links between features become stronger couplings between qubits.</Step>
            <Step n={4} t="Quantum evolution">A short counterdiabatic circuit turns those couplings into measurable correlations, giving 24 quantum features.</Step>
            <Step n={5} t="Hybrid classifier">The 8 clinical and 24 quantum features are combined and scored by a gradient boosting classifier to produce the risk probability.</Step>
          </div>
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="What the model looks at" subtitle="Clinical factors considered for every patient" icon={<ListTree className="size-4" aria-hidden />} />
        <dl className="grid gap-x-8 gap-y-3 p-5 text-sm sm:grid-cols-2">
          {Object.entries(FEATURE_RATIONALE).map(([k, v]) => (
            <div key={k} className="border-b border-line/70 pb-3">
              <dt className="font-medium text-navy-900">{v.label}</dt>
              <dd className="mt-0.5 text-ink-2">{v.text}</dd>
            </div>
          ))}
        </dl>
      </Card>
    </>
  )
}

function FactorList({ title, icon, items }: { title: string; icon: ReactNode; items: FeatureAttribution[] }) {
  return (
    <div className="rounded-lg border border-line p-3">
      <p className="flex items-center gap-1.5 text-xs font-semibold text-ink-2">{icon}{title}</p>
      {items.length ? (
        <ul className="mt-2 space-y-1 text-sm">{items.slice(0, 4).map((i) => <li key={i.feature} className="flex justify-between gap-2"><span>{i.feature}</span><span className="tabular text-ink-3">{i.value.toFixed(3)}</span></li>)}</ul>
      ) : (
        <p className="mt-2 text-xs text-ink-3">Not available yet</p>
      )}
    </div>
  )
}

function Step({ n, t, children }: { n: number; t: string; children: ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="grid size-6 shrink-0 place-items-center rounded-full bg-navy-900 text-xs font-semibold text-white">{n}</span>
      <div><p className="font-semibold text-navy-900">{t}</p><p>{children}</p></div>
    </div>
  )
}
