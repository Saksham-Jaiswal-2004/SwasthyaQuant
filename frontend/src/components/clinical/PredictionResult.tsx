import { AlertTriangle, ArrowRight, ChevronDown, FlaskConical, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'
import { DCQF_CONFIG, SERVED_MODEL } from '../../data/research'
import { deriveFeatures, GENDER_LABELS, LEVEL_LABELS } from '../../lib/clinical'
import { cx, fmtDateTime, fmtNum } from '../../lib/format'
import type { PredictionRecord } from '../../services/predictionService'
import { RiskGauge } from '../charts/RiskGauge'
import { ModelPipeline } from '../quantum/ModelPipeline'
import { SERVED_PIPELINE } from '../quantum/pipelineStages'
import { Card, Disclaimer, StatusBadge } from '../ui/primitives'

export function DemoWatermark() {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-warn/30 bg-warn-bg px-3 py-2 text-xs font-semibold uppercase tracking-wide text-warn">
      <FlaskConical className="size-4" aria-hidden />
      Demo data: illustrative placeholder, not produced by the model
    </div>
  )
}

/** Horizontal probability bar with the model's decision threshold marked. */
function ProbabilityBar({ p, tone }: { p: number; tone: 'risk' | 'good' | 'neutral' }) {
  const fill = tone === 'risk' ? 'bg-risk' : tone === 'good' ? 'bg-good' : 'bg-slate-400'
  return (
    <div className="mt-4">
      <div className="relative h-2.5 rounded-full bg-slate-200/80">
        <div className={cx('h-full rounded-full transition-[width] duration-700', fill)} style={{ width: `${Math.max(2, p * 100)}%` }} />
        <div className="absolute -top-1 h-[18px] w-0.5 rounded bg-navy-900" style={{ left: `${SERVED_MODEL.threshold * 100}%` }} aria-hidden />
      </div>
      <div className="mt-1.5 flex justify-between text-[11px] text-ink-3 tabular">
        <span>0%</span>
        <span>Decision threshold {SERVED_MODEL.threshold * 100}%</span>
        <span>100%</span>
      </div>
    </div>
  )
}

export function PredictionResult({ record }: { record: PredictionRecord }) {
  const { result, input } = record
  const demo = record.source === 'demo'
  const positive = result.prediction === 1
  const tone = demo ? 'neutral' : positive ? 'risk' : 'good'
  const d = deriveFeatures(input)
  const pct = (result.risk_probability * 100).toFixed(1)

  return (
    <Card as="article" className="overflow-hidden animate-rise">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-5 py-3.5">
        <div>
          <h2 className="text-[15px] font-semibold text-navy-900">Assessment result</h2>
          <p className="text-xs text-ink-3">{fmtDateTime(record.createdAt)} · analysed in {Math.round(record.durationMs)} ms</p>
        </div>
        <StatusBadge tone={demo ? 'warn' : 'info'}>{demo ? 'Demo output' : SERVED_MODEL.short}</StatusBadge>
      </div>

      <div className="space-y-5 p-5">
        {demo && <DemoWatermark />}

        <div className="grid items-center gap-6 md:grid-cols-[240px_minmax(0,1fr)]">
          <RiskGauge probability={result.risk_probability} predictedPositive={positive} demo={demo} />

          <div>
            <div className={cx(
              'rounded-xl border px-5 py-4',
              tone === 'risk' ? 'border-risk/20 bg-risk-bg' : tone === 'good' ? 'border-good/20 bg-good-bg' : 'border-line bg-slate-50',
            )}>
              <div className="flex items-center gap-3">
                <span className={cx('grid size-10 shrink-0 place-items-center rounded-full', tone === 'risk' ? 'bg-risk/10 text-risk' : tone === 'good' ? 'bg-good/10 text-good' : 'bg-slate-200 text-ink-2')}>
                  {positive ? <AlertTriangle className="size-5" aria-hidden /> : <ShieldCheck className="size-5" aria-hidden />}
                </span>
                <div className="min-w-0">
                  <p className={cx('text-xl font-semibold tracking-tight', tone === 'risk' ? 'text-risk' : tone === 'good' ? 'text-good' : 'text-ink-2')}>{result.risk_label}</p>
                  <p className="text-sm text-ink-2 tabular">{pct}% model-estimated probability of cardiovascular disease</p>
                </div>
              </div>
              <ProbabilityBar p={result.risk_probability} tone={tone} />
            </div>

            <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <Fact k="Model" v="Hybrid quantum" sub="+ gradient boosting" />
              <Fact k="Quantum layer" v={`${DCQF_CONFIG.qubits} qubits`} sub={`${DCQF_CONFIG.quantumFeatures} quantum features`} />
              <Fact k="Confidence" v="Not reported" sub="probability only" muted />
              <Fact k="Validated accuracy" v="Pending" sub="evaluation in progress" muted />
            </dl>
          </div>
        </div>

        <details className="group rounded-lg border border-line">
          <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-sm font-medium text-navy-900">
            How this was computed
            <ChevronDown className="size-4 text-ink-3 transition-transform group-open:rotate-180" aria-hidden />
          </summary>
          <div className="space-y-4 border-t border-line p-4">
            <ModelPipeline stages={SERVED_PIPELINE} orientation="grid" />
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
        </details>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <Disclaimer className="flex-1">This is an AI-generated risk estimate, not a medical diagnosis. Consult a clinician for medical advice.</Disclaimer>
          <Link to="/insights" className="inline-flex items-center gap-1.5 text-sm font-medium text-teal-700 hover:text-teal-600">
            View insights <ArrowRight className="size-4" aria-hidden />
          </Link>
        </div>
      </div>
    </Card>
  )
}

function Fact({ k, v, sub, muted }: { k: string; v: string; sub?: string; muted?: boolean }) {
  return (
    <div className="rounded-lg bg-slate-50 px-3 py-2">
      <dt className="text-[11px] font-medium uppercase tracking-wide text-ink-3">{k}</dt>
      <dd className={cx('mt-0.5 font-semibold', muted ? 'text-ink-3' : 'text-navy-900')}>{v}</dd>
      {sub && <dd className="text-xs text-ink-3">{sub}</dd>}
    </div>
  )
}
