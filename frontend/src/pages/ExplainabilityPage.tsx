import { ArrowDownRight, ArrowUpRight, Layers, Lightbulb, ListTree, UserRound } from 'lucide-react'
import { Link } from 'react-router-dom'
import { FeatureImportanceChart, type FeatureAttribution } from '../components/charts/FeatureImportanceChart'
import { ModelPipeline } from '../components/quantum/ModelPipeline'
import { SERVED_PIPELINE } from '../components/quantum/pipelineStages'
import { Callout, Card, CardHeader, EmptyState, PageHeader, StatusBadge } from '../components/ui/primitives'
import { FEATURE_RATIONALE } from '../data/research'
import { useAppState } from '../hooks/appStateContext'
import { deriveFeatures } from '../lib/clinical'
import { fmtDateTime, fmtNum } from '../lib/format'

/**
 * No attribution endpoint exists yet. When one is added, pass its payload here; the
 * chart and factor lists render from it unchanged.
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
      <PageHeader
        eyebrow="Explainability"
        title="Why the model says what it says"
        description="Two kinds of explanation are kept separate: SHAP explains the classical classifier over its inputs, while parameter-shift saliency is the only method that explains the quantum circuit itself."
      />

      <div className="grid gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader title="Feature importance" subtitle="Signed contribution to the predicted risk" icon={<Lightbulb className="size-4" aria-hidden />} />
          {attributions ? (
            <div className="p-5"><FeatureImportanceChart data={attributions} /></div>
          ) : (
            <EmptyState title="Attributions are not exposed by the API yet">
              SHAP tooling exists in <span className="font-mono">src/qheart/explain/shap_classical.py</span> and quantum saliency in
              <span className="font-mono"> saliency_quantum.py</span>, but no endpoint serves their output. Nothing is estimated in the browser.
            </EmptyState>
          )}
        </Card>

        <Card>
          <CardHeader title="Patient-level explanation" subtitle={latest ? `Latest assessment · ${fmtDateTime(latest.createdAt)}` : 'Run an assessment first'} icon={<UserRound className="size-4" aria-hidden />} />
          <div className="p-5">
            {!latest ? (
              <EmptyState title="No assessment to explain" action={<Link to="/app/assess" className="text-sm font-medium text-teal-700 hover:text-teal-600">Go to Disease Assessment →</Link>} />
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <StatusBadge tone={latest.result.prediction ? 'risk' : 'good'}>{latest.result.risk_label}</StatusBadge>
                  <span className="text-sm text-ink-2 tabular">{(latest.result.risk_probability * 100).toFixed(1)}% model-predicted risk</span>
                  {latest.source === 'demo' && <StatusBadge tone="warn">Demo data</StatusBadge>}
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  <FactorList title="Factors increasing predicted risk" icon={<ArrowUpRight className="size-4 text-risk" aria-hidden />} items={up} />
                  <FactorList title="Factors reducing predicted risk" icon={<ArrowDownRight className="size-4 text-navy-600" aria-hidden />} items={down} />
                </div>
                <div className="rounded-lg bg-slate-50 p-3 text-sm">
                  <p className="text-xs font-medium uppercase tracking-wide text-ink-3">Derived features the pipeline computed</p>
                  <p className="mt-1 text-ink-2 tabular">
                    BMI {fmtNum(d?.bmi)} ({d?.bmiClass ?? '—'}) · pulse pressure {fmtNum(d?.pulsePressure, 0)} mmHg · MAP {fmtNum(d?.map, 0)} mmHg · BP group {d?.bpGroup ?? '—'}
                  </p>
                  <p className="mt-1 text-xs text-ink-3">These are the model&rsquo;s inputs, not an attribution of the result.</p>
                </div>
              </div>
            )}
          </div>
        </Card>
      </div>

      <Card className="mt-5">
        <CardHeader title="How the hybrid model reaches a prediction" subtitle="Classical preprocessing → quantum feature map → classical classifier" icon={<Layers className="size-4" aria-hidden />} />
        <div className="grid gap-6 p-5 lg:grid-cols-[300px_minmax(0,1fr)]">
          <ModelPipeline stages={SERVED_PIPELINE} orientation="vertical" />
          <div className="space-y-4 text-sm leading-relaxed text-ink-2">
            <Step n={1} t="Clinical preprocessing">Impossible measurements become missing values. BMI, pulse pressure and mean arterial pressure are derived. Imputation statistics come from training data only.</Step>
            <Step n={2} t="Feature selection">Eight features are chosen by mutual information with a redundancy penalty (MI-mRMR, k = 8), fitted inside each training fold.</Step>
            <Step n={3} t="Quantum feature encoding">Each selected feature becomes a rotation angle on its own qubit. Mutual information between feature pairs sets the strength of the qubit couplings.</Step>
            <Step n={4} t="Counterdiabatic evolution">One digitized step of a counterdiabatic sweep turns the encoded fields into measurable correlations. Z-string expectations give 24 quantum features.</Step>
            <Step n={5} t="Classical classifier">The 8 classical and 24 quantum features are standardised and passed to a Gradient Boosting classifier, which outputs the risk probability.</Step>
          </div>
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Candidate clinical features" subtitle="Why each variable is considered (src/qheart/features/select.py)" icon={<ListTree className="size-4" aria-hidden />} />
        <dl className="grid gap-x-8 gap-y-3 p-5 text-sm sm:grid-cols-2">
          {Object.entries(FEATURE_RATIONALE).map(([k, v]) => (
            <div key={k} className="border-b border-line/70 pb-3">
              <dt className="font-medium text-navy-900">{v.label} <span className="font-mono text-xs font-normal text-ink-3">{k}</span></dt>
              <dd className="mt-0.5 text-ink-2">{v.text}</dd>
            </div>
          ))}
        </dl>
      </Card>

      <Callout tone="info" className="mt-5" title="Reading SHAP on a hybrid model">
        SHAP over the hybrid classifier explains the classifier&rsquo;s view of its 32 inputs, including quantum expectation values. It
        explains the model, not the disease, and says nothing about the circuit&rsquo;s internals.
      </Callout>
    </>
  )
}

function FactorList({ title, icon, items }: { title: string; icon: React.ReactNode; items: FeatureAttribution[] }) {
  return (
    <div className="rounded-lg border border-line p-3">
      <p className="flex items-center gap-1.5 text-xs font-semibold text-ink-2">{icon}{title}</p>
      {items.length ? (
        <ul className="mt-2 space-y-1 text-sm">{items.slice(0, 4).map((i) => <li key={i.feature} className="flex justify-between gap-2"><span>{i.feature}</span><span className="tabular text-ink-3">{i.value.toFixed(3)}</span></li>)}</ul>
      ) : (
        <p className="mt-2 text-xs text-ink-3">Requires an attribution endpoint. Not available.</p>
      )}
    </div>
  )
}

function Step({ n, t, children }: { n: number; t: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="grid size-6 shrink-0 place-items-center rounded-full bg-navy-900 text-xs font-semibold text-white">{n}</span>
      <div><p className="font-semibold text-navy-900">{t}</p><p>{children}</p></div>
    </div>
  )
}
