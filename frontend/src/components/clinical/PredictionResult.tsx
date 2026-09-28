import { ArrowRight, FlaskConical } from 'lucide-react'
import { Link } from 'react-router-dom'
import { SERVED_MODEL } from '../../data/research'
import { deriveFeatures, GENDER_LABELS, LEVEL_LABELS } from '../../lib/clinical'
import { cx, fmtDateTime, fmtNum } from '../../lib/format'
import type { PredictionRecord } from '../../services/predictionService'
import { RiskGauge } from '../charts/RiskGauge'
import { ModelPipeline } from '../quantum/ModelPipeline'
import { SERVED_PIPELINE } from '../quantum/pipelineStages'
import { Callout, Card, Disclaimer, StatusBadge } from '../ui/primitives'

export function DemoWatermark() {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-warn/30 bg-warn-bg px-3 py-2 text-xs font-semibold uppercase tracking-wide text-warn">
      <FlaskConical className="size-4" aria-hidden />
      Demo data: illustrative placeholder, not produced by the model
    </div>
  )
}

export function PredictionResult({ record }: { record: PredictionRecord }) {
  const { result, input } = record
  const demo = record.source === 'demo'
  const positive = result.prediction === 1
  const d = deriveFeatures(input)

  return (
    <Card as="article" className="overflow-hidden animate-rise">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-navy-50/50 px-5 py-3">
        <div>
          <h2 className="text-[15px] font-semibold text-navy-900">Disease Risk Assessment</h2>
          <p className="text-xs text-ink-3">{fmtDateTime(record.createdAt)} · response in {Math.round(record.durationMs)} ms</p>
        </div>
        <StatusBadge tone={demo ? 'warn' : 'info'}>{demo ? 'Demo output' : 'Returned by /api/predict'}</StatusBadge>
      </div>

      <div className="space-y-5 p-5">
        {demo && <DemoWatermark />}

        <div className="grid items-center gap-6 md:grid-cols-[280px_1fr]">
          <RiskGauge probability={result.risk_probability} predictedPositive={positive} demo={demo} />
          <div>
            <p className="text-xs font-medium uppercase tracking-wide text-ink-3">Model prediction</p>
            <p className={cx('mt-1 text-2xl font-semibold tracking-tight', demo ? 'text-ink-2' : positive ? 'text-risk' : 'text-good')}>
              {result.risk_label}
            </p>
            <p className="mt-1 text-sm text-ink-2">
              Predicted class <span className="font-mono font-semibold">{result.prediction}</span> at the model&rsquo;s {SERVED_MODEL.threshold} probability threshold
              (<span className="tabular">{result.risk_probability.toFixed(4)}</span>).
            </p>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <div className="rounded-lg bg-slate-50 px-3 py-2">
                <dt className="text-xs text-ink-3">Model used</dt>
                <dd className="font-medium text-navy-900">{SERVED_MODEL.short}</dd>
                <dd className="text-xs text-ink-3">DCQF quantum features + Gradient Boosting</dd>
              </div>
              <div className="rounded-lg bg-slate-50 px-3 py-2">
                <dt className="text-xs text-ink-3">Confidence</dt>
                <dd className="font-medium text-ink-2">Not provided</dd>
                <dd className="text-xs text-ink-3">The API returns a probability only</dd>
              </div>
            </dl>
          </div>
        </div>

        <Callout tone="warn" title="Reference accuracy for this model is not available yet">
          The hybrid model has no rows in the research ledger, so its sensitivity and specificity cannot be shown beside this estimate.
          Classical baselines are on the <Link to="/app/benchmarks" className="font-medium underline underline-offset-2">Benchmarks</Link> page.
        </Callout>

        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-3">Processing pipeline</p>
          <ModelPipeline stages={SERVED_PIPELINE} compact />
        </div>

        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-ink-3">Inputs and derived clinical features</p>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-4">
            {[
              ['Age', `${input.age} y`], ['Gender', GENDER_LABELS[input.gender]],
              ['Blood pressure', `${input.ap_hi}/${input.ap_lo} mmHg`], ['BMI', `${fmtNum(d.bmi)} (${d.bmiClass ?? '—'})`],
              ['Pulse pressure', `${fmtNum(d.pulsePressure, 0)} mmHg`], ['Mean arterial', `${fmtNum(d.map, 0)} mmHg`],
              ['Cholesterol', LEVEL_LABELS[input.cholesterol]], ['Glucose', LEVEL_LABELS[input.gluc]],
            ].map(([k, v]) => (
              <div key={k} className="min-w-0">
                <dt className="text-xs text-ink-3">{k}</dt>
                <dd className="truncate font-medium text-ink tabular">{v}</dd>
              </div>
            ))}
          </dl>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <Disclaimer className="flex-1">
            This result is an AI-generated risk estimate. {result.disclaimer}
          </Disclaimer>
          <Link to="/app/explain" className="inline-flex items-center gap-1.5 text-sm font-medium text-teal-700 hover:text-teal-600">
            Explain this result <ArrowRight className="size-4" aria-hidden />
          </Link>
        </div>
      </div>
    </Card>
  )
}
