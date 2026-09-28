import { Activity, HeartPulse, Loader2, RotateCcw, ScanHeart, Sparkles, User } from 'lucide-react'
import { useMemo, useRef, useState, type FormEvent } from 'react'
import { ChoiceField, FieldGroup, NumberField } from '../components/clinical/Fields'
import { PredictionResult } from '../components/clinical/PredictionResult'
import { Button, Callout, Card, Disclaimer, PageHeader, StatusBadge } from '../components/ui/primitives'
import { DEMO_MODE } from '../config'
import { useAppState } from '../hooks/appStateContext'
import { ApiError } from '../lib/api'
import { deriveFeatures } from '../lib/clinical'
import { fmtNum } from '../lib/format'
import { predict, type PredictionRecord } from '../services/predictionService'
import type { PatientInput } from '../types/api'

type FormState = { [K in keyof PatientInput]: PatientInput[K] | '' }
type Errors = Partial<Record<keyof PatientInput, string>>

/** The example payload documented in backend/README.md. */
const EXAMPLE: PatientInput = { age: 58, height: 170, weight: 78, ap_hi: 145, ap_lo: 90, smoke: 0, alco: 0, active: 1, gender: 1, cholesterol: 1, gluc: 1 }
const EMPTY: FormState = { age: '', height: '', weight: '', ap_hi: '', ap_lo: '', smoke: 0, alco: 0, active: 1, gender: '', cholesterol: 1, gluc: 1 }

// Ranges from backend/app/schemas/patient.py and src/qheart/schema.py (PLAUSIBLE).
const RANGES = { age: [18, 120], height: [100, 250], weight: [20, 300], ap_hi: [60, 250], ap_lo: [30, 150] } as const

function validate(f: FormState): Errors {
  const e: Errors = {}
  for (const [k, [lo, hi]] of Object.entries(RANGES) as [keyof typeof RANGES, readonly [number, number]][]) {
    const v = f[k]
    if (v === '') e[k] = 'Required'
    else if (!Number.isFinite(v) || v < lo || v > hi) e[k] = `Must be between ${lo} and ${hi}`
    else if (k !== 'weight' && !Number.isInteger(v)) e[k] = 'Whole numbers only'
  }
  if (!e.ap_hi && !e.ap_lo && typeof f.ap_hi === 'number' && typeof f.ap_lo === 'number' && f.ap_hi < f.ap_lo) {
    e.ap_hi = 'Systolic must be greater than or equal to diastolic'
  }
  if (f.gender === '') e.gender = 'Select one'
  return e
}

type Submit =
  | { state: 'idle' }
  | { state: 'running' }
  | { state: 'error'; error: ApiError | Error }

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

  const set = <K extends keyof PatientInput>(k: K) => (v: PatientInput[K] | '') => {
    setForm((f) => ({ ...f, [k]: v }))
    setServerErrors((s) => ({ ...s, [k]: undefined }))
  }

  async function onSubmit(ev: FormEvent) {
    ev.preventDefault()
    setTouched(true)
    if (Object.keys(clientErrors).length) {
      document.querySelector<HTMLElement>('[aria-invalid="true"]')?.focus()
      return
    }
    setSubmit({ state: 'running' })
    try {
      const rec = await predict(form as PatientInput)
      setRecord(rec)
      addRecord(rec)
      setSubmit({ state: 'idle' })
      requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
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
  const yesNo = [{ value: 0 as const, label: 'No' }, { value: 1 as const, label: 'Yes' }]
  const levels = [{ value: 1 as const, label: 'Normal' }, { value: 2 as const, label: 'Above normal' }, { value: 3 as const, label: 'Well above' }]

  return (
    <>
      <PageHeader
        eyebrow="Disease Assessment"
        title="Cardiovascular risk assessment"
        description="Enter the 11 clinical parameters the hybrid quantum-classical model was trained on. Values are validated against the same ranges the API enforces."
        actions={
          <>
            <Button variant="secondary" size="sm" type="button" onClick={() => { setForm(EXAMPLE); setServerErrors({}) }}>
              <Sparkles className="size-4" aria-hidden /> Load example
            </Button>
            <Button variant="ghost" size="sm" type="button" onClick={() => { setForm(EMPTY); setTouched(false); setServerErrors({}); setRecord(null); setSubmit({ state: 'idle' }) }}>
              <RotateCcw className="size-4" aria-hidden /> Reset
            </Button>
          </>
        }
      />

      {service.state === 'online' && !service.health.model_loaded && (
        <Callout tone="warn" className="mb-5" title="The API is online, but no trained model is loaded">
          {DEMO_MODE
            ? 'Demo mode is on: submissions will show a clearly labelled illustrative placeholder instead of a model output.'
            : 'Submissions will return "model not loaded" until a frozen inference bundle is placed in backend/artifacts/.'}
        </Callout>
      )}
      {service.state === 'offline' && (
        <Callout tone="risk" className="mb-5" title="Inference API unreachable">
          Start the FastAPI backend on port 8000 (see README). {DEMO_MODE && 'Demo mode will show illustrative output meanwhile.'}
        </Callout>
      )}

      <form onSubmit={onSubmit} noValidate className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-5">
          <FieldGroup title="Patient information" description="Demographics and body measurements" icon={<User className="size-4" aria-hidden />}>
            <NumberField label="Age" unit="years" {...rng('age')} value={form.age} onChange={set('age')} error={errors.age} help="Converted to days by the API to match the training data." hint="18–120 years · training cohort mostly 30–65" />
            <ChoiceField label="Gender" value={form.gender === '' ? undefined : form.gender} onChange={set('gender')} options={[{ value: 1, label: 'Female' }, { value: 2, label: 'Male' }]} error={errors.gender} hint="Dataset coding: 1 = female, 2 = male" />
            <NumberField label="Height" unit="cm" {...rng('height')} value={form.height} onChange={set('height')} error={errors.height} />
            <NumberField label="Weight" unit="kg" {...rng('weight')} step={0.1} value={form.weight} onChange={set('weight')} error={errors.weight} />
          </FieldGroup>

          <FieldGroup title="Blood pressure" description="Resting measurement" icon={<HeartPulse className="size-4" aria-hidden />}>
            <NumberField label="Systolic (ap_hi)" unit="mmHg" {...rng('ap_hi')} value={form.ap_hi} onChange={set('ap_hi')} error={errors.ap_hi} help="Upper blood-pressure reading. Must be at least the diastolic value." />
            <NumberField label="Diastolic (ap_lo)" unit="mmHg" {...rng('ap_lo')} value={form.ap_lo} onChange={set('ap_lo')} error={errors.ap_lo} help="Lower blood-pressure reading." />
          </FieldGroup>

          <FieldGroup title="Laboratory markers" description="Categorical levels as recorded in the dataset" icon={<ScanHeart className="size-4" aria-hidden />}>
            <ChoiceField label="Cholesterol" value={form.cholesterol === '' ? undefined : form.cholesterol} onChange={set('cholesterol')} options={levels} />
            <ChoiceField label="Glucose" value={form.gluc === '' ? undefined : form.gluc} onChange={set('gluc')} options={levels} />
          </FieldGroup>

          <FieldGroup title="Lifestyle factors" description="Self-reported" icon={<Activity className="size-4" aria-hidden />}>
            <ChoiceField label="Smoker" value={form.smoke === '' ? undefined : form.smoke} onChange={set('smoke')} options={yesNo} />
            <ChoiceField label="Alcohol intake" value={form.alco === '' ? undefined : form.alco} onChange={set('alco')} options={yesNo} />
            <ChoiceField label="Physically active" value={form.active === '' ? undefined : form.active} onChange={set('active')} options={yesNo} />
          </FieldGroup>
        </div>

        <aside className="lg:sticky lg:top-24 lg:self-start">
          <Card className="p-5">
            <p className="text-sm font-semibold text-navy-900">Derived clinical features</p>
            <p className="mt-0.5 text-xs text-ink-3">Computed with the same formulas as <span className="font-mono">features/clinical.py</span>. Display only; the backend recomputes them.</p>
            <dl className="mt-4 space-y-2.5 text-sm">
              <Derived label="Body-mass index" value={fmtNum(derived.bmi)} unit="kg/m²" tag={derived.bmiClass} />
              <Derived label="Pulse pressure" value={fmtNum(derived.pulsePressure, 0)} unit="mmHg" />
              <Derived label="Mean arterial pressure" value={fmtNum(derived.map, 1)} unit="mmHg" />
              <Derived label="BP group (systolic)" value={derived.bpGroup ?? '—'} />
              <Derived label="Age group" value={derived.ageGroup ?? '—'} />
            </dl>

            <Button type="submit" size="lg" className="mt-5 w-full" disabled={running} aria-describedby="analyze-status">
              {running ? <><Loader2 className="size-5 animate-spin" aria-hidden /> Running Hybrid Quantum Analysis…</> : 'Analyze Risk'}
            </Button>
            <p id="analyze-status" className="sr-only" aria-live="polite">{running ? 'Analyzing clinical data' : ''}</p>
            {touched && Object.keys(clientErrors).length > 0 && (
              <p className="mt-2 text-center text-xs text-risk">Please correct {Object.keys(clientErrors).length} highlighted field{Object.keys(clientErrors).length > 1 ? 's' : ''}.</p>
            )}
            <Disclaimer className="mt-4" />
          </Card>
        </aside>
      </form>

      <div ref={resultRef} className="scroll-mt-24 pt-6">
        {submit.state === 'error' && <SubmitError error={submit.error} />}
        {record && submit.state !== 'error' && <PredictionResult record={record} />}
      </div>
    </>
  )
}

const rng = (k: keyof typeof RANGES) => ({ min: RANGES[k][0], max: RANGES[k][1] })

function Derived({ label, value, unit, tag }: { label: string; value: string; unit?: string; tag?: string | null }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-line/70 pb-2 last:border-0">
      <dt className="text-ink-2">{label}</dt>
      <dd className="flex items-center gap-2 text-right">
        <span className="font-semibold text-navy-900 tabular">{value}</span>
        {unit && value !== '—' && <span className="text-xs text-ink-3">{unit}</span>}
        {tag && <StatusBadge tone="neutral">{tag}</StatusBadge>}
      </dd>
    </div>
  )
}

function SubmitError({ error }: { error: Error }) {
  const e = error instanceof ApiError ? error : null
  const title =
    e?.kind === 'unavailable' ? 'Prediction model not loaded'
    : e?.kind === 'network' ? 'Unable to reach the inference service'
    : e?.kind === 'validation' ? 'Some values were rejected by the API'
    : 'Unable to complete analysis. Please try again.'
  const body =
    e?.kind === 'unavailable' ? 'The API is running but has no trained inference bundle. No risk estimate was produced, and none is shown.'
    : e?.kind === 'network' ? 'Check that the FastAPI backend is running on port 8000, then try again.'
    : e?.kind === 'validation' ? 'The highlighted fields above show what the API rejected.'
    : 'The service returned an unexpected response.'
  return (
    <Card className="p-5 animate-rise" as="div">
      <Callout tone={e?.kind === 'validation' ? 'warn' : 'risk'} title={title}>
        <p>{body}</p>
        {e?.detail && (
          <details className="mt-2">
            <summary className="cursor-pointer text-xs font-medium">Technical detail from the API</summary>
            <p className="mt-1 text-xs opacity-90">{e.detail}</p>
          </details>
        )}
      </Callout>
    </Card>
  )
}
