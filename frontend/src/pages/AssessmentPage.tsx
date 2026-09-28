import { Activity, Droplets, HeartPulse, Loader2, RotateCcw, Ruler, Sparkles, Stethoscope, User } from 'lucide-react'
import { useMemo, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { ChoiceField, NumberField, PanelSection, SwitchField } from '../components/clinical/Fields'
import { PredictionResult } from '../components/clinical/PredictionResult'
import { ModelPipeline } from '../components/quantum/ModelPipeline'
import { SUMMARY_PIPELINE } from '../components/quantum/pipelineStages'
import { Button, Callout, Card, StatusBadge } from '../components/ui/primitives'
import { DEMO_MODE } from '../config'
import { useAppState } from '../hooks/appStateContext'
import { ApiError } from '../lib/api'
import { deriveFeatures, type DerivedFeatures } from '../lib/clinical'
import { cx, fmtNum } from '../lib/format'
import { predict, type PredictionRecord } from '../services/predictionService'
import type { PatientInput } from '../types/api'

type FormState = { [K in keyof PatientInput]: PatientInput[K] | '' }
type Errors = Partial<Record<keyof PatientInput, string>>

const EXAMPLE: PatientInput = { age: 58, height: 170, weight: 78, ap_hi: 145, ap_lo: 90, smoke: 0, alco: 0, active: 1, gender: 1, cholesterol: 1, gluc: 1 }
const EMPTY: FormState = { age: '', height: '', weight: '', ap_hi: '', ap_lo: '', smoke: 0, alco: 0, active: 1, gender: '', cholesterol: 1, gluc: 1 }

// Ranges from backend/app/schemas/patient.py and src/qheart/schema.py (PLAUSIBLE).
const RANGES = { age: [18, 120], height: [100, 250], weight: [20, 300], ap_hi: [60, 250], ap_lo: [30, 150] } as const
const rng = (k: keyof typeof RANGES) => ({ min: RANGES[k][0], max: RANGES[k][1] })

function validate(f: FormState): Errors {
  const e: Errors = {}
  for (const [k, [lo, hi]] of Object.entries(RANGES) as [keyof typeof RANGES, readonly [number, number]][]) {
    const v = f[k]
    if (v === '') e[k] = 'Required'
    else if (!Number.isFinite(v) || v < lo || v > hi) e[k] = `Enter ${lo}–${hi}`
    else if (k !== 'weight' && !Number.isInteger(v)) e[k] = 'Whole numbers only'
  }
  if (!e.ap_hi && !e.ap_lo && typeof f.ap_hi === 'number' && typeof f.ap_lo === 'number' && f.ap_hi < f.ap_lo) {
    e.ap_hi = 'Must be ≥ diastolic'
  }
  if (f.gender === '') e.gender = 'Select one'
  return e
}

type Submit = { state: 'idle' } | { state: 'running' } | { state: 'error'; error: ApiError | Error }

const levels = [{ value: 1 as const, label: 'Normal' }, { value: 2 as const, label: 'Above' }, { value: 3 as const, label: 'Well above' }]

export function AssessmentPage() {
  const { addRecord, service } = useAppState()
  const [form, setForm] = useState<FormState>(EXAMPLE)
  const [touched, setTouched] = useState(false)
  const [serverErrors, setServerErrors] = useState<Errors>({})
  const [submit, setSubmit] = useState<Submit>({ state: 'idle' })
  const [record, setRecord] = useState<PredictionRecord | null>(null)
  const resultRef = useRef<HTMLDivElement>(null)

  const clientErrors = useMemo(() => validate(form), [form])
  const errors: Errors = touched ? { ...clientErrors, ...serverErrors } : serverErrors
  const derived = deriveFeatures(Object.fromEntries(Object.entries(form).filter(([, v]) => v !== '')) as Partial<PatientInput>)
  const errorCount = Object.keys(clientErrors).length

  const set = <K extends keyof PatientInput>(k: K) => (v: PatientInput[K] | '') => {
    setForm((f) => ({ ...f, [k]: v }))
    setServerErrors((s) => ({ ...s, [k]: undefined }))
  }
  const flag = (k: 'smoke' | 'alco' | 'active') => (on: boolean) => set(k)(on ? 1 : 0)

  async function onSubmit(ev: FormEvent) {
    ev.preventDefault()
    setTouched(true)
    if (errorCount) {
      document.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus()
      return
    }
    setSubmit({ state: 'running' })
    requestAnimationFrame(() => {
      if (window.innerWidth < 1024) resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    })
    try {
      const rec = await predict(form as PatientInput)
      setRecord(rec)
      addRecord(rec)
      setSubmit({ state: 'idle' })
    } catch (err) {
      if (err instanceof ApiError && err.kind === 'validation') {
        const mapped: Errors = {}
        for (const issue of err.issues ?? []) {
          const field = issue.loc[issue.loc.length - 1] as keyof PatientInput
          mapped[field] = issue.msg.replace(/^Value error, /, '')
        }
        setServerErrors(mapped)
      }
      setSubmit({ state: 'error', error: err as Error })
    }
  }

  const running = submit.state === 'running'
  const engineDown = service.state === 'offline' || (service.state === 'online' && !service.health.model_loaded)

  return (
    <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
      <PageIntro className="lg:hidden" />
      {/* Patient profile panel */}
      <form onSubmit={onSubmit} noValidate aria-label="Patient profile" className="lg:sticky lg:top-[5.5rem] lg:self-start">
        <Card as="div" className="flex flex-col overflow-hidden lg:max-h-[calc(100dvh-7rem)]">
          <div className="flex items-center justify-between gap-2 border-b border-line px-5 py-3.5">
            <div>
              <p className="text-[15px] font-semibold text-navy-900">Patient profile</p>
              <p className="text-xs text-ink-3">11 routine health metrics</p>
            </div>
            <div className="flex gap-1">
              <button type="button" onClick={() => { setForm(EXAMPLE); setServerErrors({}) }} className="grid size-8 place-items-center rounded-md text-ink-3 hover:bg-slate-100 hover:text-navy-900" title="Load sample patient" aria-label="Load sample patient">
                <Sparkles className="size-4" aria-hidden />
              </button>
              <button type="button" onClick={() => { setForm(EMPTY); setTouched(false); setServerErrors({}); setRecord(null); setSubmit({ state: 'idle' }) }} className="grid size-8 place-items-center rounded-md text-ink-3 hover:bg-slate-100 hover:text-navy-900" title="Clear form" aria-label="Clear form">
                <RotateCcw className="size-4" aria-hidden />
              </button>
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto">
            <PanelSection title="Demographics" icon={<User className="size-3.5" aria-hidden />}>
              <ChoiceField label="Gender" value={form.gender === '' ? undefined : form.gender} onChange={set('gender')} options={[{ value: 1, label: 'Female' }, { value: 2, label: 'Male' }]} error={errors.gender} />
              <NumberField label="Age" unit="years" {...rng('age')} value={form.age} onChange={set('age')} error={errors.age} />
            </PanelSection>

            <PanelSection title="Body measurements" icon={<Ruler className="size-3.5" aria-hidden />}>
              <div className="grid grid-cols-2 gap-3">
                <NumberField label="Height" unit="cm" {...rng('height')} value={form.height} onChange={set('height')} error={errors.height} />
                <NumberField label="Weight" unit="kg" {...rng('weight')} step={0.1} value={form.weight} onChange={set('weight')} error={errors.weight} />
              </div>
            </PanelSection>

            <PanelSection title="Blood pressure" icon={<HeartPulse className="size-3.5" aria-hidden />}>
              <div className="grid grid-cols-2 gap-3">
                <NumberField label="Systolic" unit="mmHg" {...rng('ap_hi')} value={form.ap_hi} onChange={set('ap_hi')} error={errors.ap_hi} help="Upper reading, measured at rest." />
                <NumberField label="Diastolic" unit="mmHg" {...rng('ap_lo')} value={form.ap_lo} onChange={set('ap_lo')} error={errors.ap_lo} help="Lower reading, measured at rest." />
              </div>
            </PanelSection>

            <PanelSection title="Lab markers" icon={<Droplets className="size-3.5" aria-hidden />}>
              <ChoiceField label="Cholesterol" value={form.cholesterol === '' ? undefined : form.cholesterol} onChange={set('cholesterol')} options={levels} />
              <ChoiceField label="Glucose" value={form.gluc === '' ? undefined : form.gluc} onChange={set('gluc')} options={levels} />
            </PanelSection>

            <PanelSection title="Lifestyle" icon={<Activity className="size-3.5" aria-hidden />} className="[&>div]:space-y-0">
              <SwitchField label="Physically active" description="Regular physical activity" checked={form.active === 1} onChange={flag('active')} />
              <SwitchField label="Smoker" checked={form.smoke === 1} onChange={flag('smoke')} />
              <SwitchField label="Drinks alcohol" checked={form.alco === 1} onChange={flag('alco')} />
            </PanelSection>
          </div>

          <div className="border-t border-line bg-slate-50/70 p-4">
            <Button type="submit" size="lg" className="w-full" disabled={running}>
              {running ? <><Loader2 className="size-5 animate-spin" aria-hidden /> Running Hybrid Quantum Analysis…</> : <><Stethoscope className="size-5" aria-hidden /> Analyze Risk</>}
            </Button>
            <p className="sr-only" aria-live="polite">{running ? 'Analyzing clinical data' : ''}</p>
            {touched && errorCount > 0 && (
              <p className="mt-2 text-center text-xs font-medium text-risk">Check {errorCount} highlighted field{errorCount > 1 ? 's' : ''}</p>
            )}
          </div>
        </Card>
      </form>

      {/* Results workspace */}
      <div className="min-w-0 space-y-5">
        <PageIntro className="hidden lg:block" />

        {engineDown && (
          <Callout tone={service.state === 'offline' ? 'risk' : 'warn'} title={service.state === 'offline' ? 'Risk engine offline' : 'Risk engine unavailable'}>
            {service.state === 'offline'
              ? 'The assessment service cannot be reached right now.'
              : 'The prediction model is not loaded, so assessments cannot be completed yet.'}
            {DEMO_MODE && ' Demo mode will show a labelled illustrative result.'}
          </Callout>
        )}

        <HealthMetrics d={derived} bp={typeof form.ap_hi === 'number' && typeof form.ap_lo === 'number' ? `${form.ap_hi}/${form.ap_lo}` : '—'} />

        <div ref={resultRef} className="scroll-mt-24">
          {running ? (
            <Card className="p-8 animate-fade">
              <div className="flex flex-col items-center text-center" role="status">
                <Loader2 className="size-8 animate-spin text-teal-600" aria-hidden />
                <p className="mt-3 font-semibold text-navy-900">Analyzing clinical data…</p>
                <p className="mt-1 text-sm text-ink-3">Encoding features into the quantum circuit and scoring with the hybrid classifier.</p>
              </div>
            </Card>
          ) : submit.state === 'error' ? (
            <SubmitError error={submit.error} />
          ) : record ? (
            <PredictionResult record={record} />
          ) : (
            <Card className="p-6">
              <div className="flex flex-col items-center py-4 text-center">
                <span className="grid size-12 place-items-center rounded-full bg-teal-50 text-teal-700"><Stethoscope className="size-6" aria-hidden /></span>
                <p className="mt-3 font-semibold text-navy-900">Ready to assess</p>
                <p className="mt-1 max-w-md text-sm text-ink-3">Review the patient profile, then select <span className="font-medium text-ink-2">Analyze Risk</span>. The estimate appears here.</p>
              </div>
              <div className="mt-4 border-t border-line pt-5">
                <p className="mb-3 text-xs font-medium uppercase tracking-wide text-ink-3">How the estimate is produced</p>
                <ModelPipeline stages={SUMMARY_PIPELINE} />
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

function PageIntro({ className }: { className?: string }) {
  return (
    <header className={cx('animate-rise', className)}>
      <h1 className="text-2xl font-semibold tracking-tight text-navy-900 sm:text-[28px]">Cardiovascular Risk Assessment</h1>
      <p className="mt-1.5 text-[15px] text-ink-2">An AI risk estimate from routine health metrics, powered by a hybrid quantum-classical model.</p>
    </header>
  )
}

const bmiTone = (c: DerivedFeatures['bmiClass']) => (c === 'Normal' ? 'good' : c === 'Obese' ? 'risk' : c ? 'warn' : 'neutral')
const bpTone = (g: DerivedFeatures['bpGroup']) => (g === 'Normal' ? 'good' : g === 'Stage 2' || g === 'Crisis' ? 'risk' : g ? 'warn' : 'neutral')

function HealthMetrics({ d, bp }: { d: DerivedFeatures; bp: string }) {
  return (
    <section aria-label="Health metrics" className="grid grid-cols-2 gap-3 xl:grid-cols-4">
      <Metric label="Body-mass index" value={fmtNum(d.bmi)} unit="kg/m²" badge={d.bmiClass && <StatusBadge tone={bmiTone(d.bmiClass)}>{d.bmiClass}</StatusBadge>} />
      <Metric label="Blood pressure" value={bp} unit="mmHg" badge={d.bpGroup && <StatusBadge tone={bpTone(d.bpGroup)}>{d.bpGroup}</StatusBadge>} />
      <Metric label="Pulse pressure" value={fmtNum(d.pulsePressure, 0)} unit="mmHg" />
      <Metric label="Mean arterial pressure" value={fmtNum(d.map, 0)} unit="mmHg" />
    </section>
  )
}

function Metric({ label, value, unit, badge }: { label: string; value: string; unit: string; badge?: ReactNode }) {
  return (
    <Card as="div" className="p-4">
      <p className="text-xs font-medium text-ink-3">{label}</p>
      <p className={cx('mt-1.5 text-2xl font-semibold tracking-tight tabular', value === '—' ? 'text-ink-3' : 'text-navy-900')}>
        {value}{value !== '—' && <span className="ml-1 text-xs font-medium text-ink-3">{unit}</span>}
      </p>
      <div className="mt-1.5 h-5">{badge}</div>
    </Card>
  )
}

function SubmitError({ error }: { error: Error }) {
  const e = error instanceof ApiError ? error : null
  const title =
    e?.kind === 'unavailable' ? 'Assessment could not be completed'
    : e?.kind === 'network' ? 'Unable to reach the risk engine'
    : e?.kind === 'validation' ? 'Please review the patient profile'
    : 'Unable to complete analysis. Please try again.'
  const body =
    e?.kind === 'unavailable' ? 'The prediction model is not loaded. No risk estimate was produced.'
    : e?.kind === 'network' ? 'Check your connection to the assessment service and try again.'
    : e?.kind === 'validation' ? 'Some values were outside the accepted range. The fields are highlighted.'
    : 'The service returned an unexpected response.'
  return (
    <Card className="p-5 animate-rise" as="div">
      <Callout tone={e?.kind === 'validation' ? 'warn' : 'risk'} title={title}>
        <p>{body}</p>
        {e?.detail && (
          <details className="mt-2">
            <summary className="cursor-pointer text-xs font-medium">Technical details</summary>
            <p className="mt-1 text-xs opacity-90">{e.detail}</p>
          </details>
        )}
      </Callout>
    </Card>
  )
}
