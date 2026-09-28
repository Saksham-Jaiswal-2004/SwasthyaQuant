import { useId, type ReactNode } from 'react'
import { cx } from '../../lib/format'
import { InfoTip, Segmented } from '../ui/primitives'

function FieldShell({ id, label, hint, error, required, help, children }: { id: string; label: string; hint?: ReactNode; error?: string; required?: boolean; help?: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <div className="mb-1.5 flex items-center gap-1.5">
        <label htmlFor={id} className="text-sm font-medium text-ink">
          {label}
          {required && <span className="ml-0.5 text-risk" aria-hidden>*</span>}
          {required && <span className="sr-only"> (required)</span>}
        </label>
        {help && <InfoTip text={help} label={`About ${label}`} />}
      </div>
      {children}
      {error ? (
        <p id={`${id}-err`} className="mt-1 text-xs font-medium text-risk" role="alert">{error}</p>
      ) : hint ? (
        <p id={`${id}-hint`} className="mt-1 text-xs text-ink-3">{hint}</p>
      ) : null}
    </div>
  )
}

export function NumberField({ label, value, onChange, unit, min, max, step = 1, hint, error, help, required = true }: {
  label: string; value: number | ''; onChange: (v: number | '') => void; unit: string; min: number; max: number; step?: number; hint?: ReactNode; error?: string; help?: string; required?: boolean
}) {
  const id = useId()
  return (
    <FieldShell id={id} label={label} hint={hint ?? `${min}–${max} ${unit}`} error={error} required={required} help={help}>
      <div className={cx('flex h-10 items-center rounded-lg border bg-surface transition-colors focus-within:border-teal-600 focus-within:ring-2 focus-within:ring-teal-600/15', error ? 'border-risk/60' : 'border-line')}>
        <input
          id={id}
          type="number"
          inputMode="decimal"
          min={min}
          max={max}
          step={step}
          required={required}
          value={value}
          aria-invalid={!!error}
          aria-describedby={error ? `${id}-err` : `${id}-hint`}
          onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
          className="h-full w-full min-w-0 rounded-l-lg bg-transparent px-3 text-[15px] text-ink tabular outline-none"
        />
        <span className="shrink-0 border-l border-line px-3 text-xs font-medium text-ink-3">{unit}</span>
      </div>
    </FieldShell>
  )
}

export function ChoiceField<T extends number>({ label, value, onChange, options, hint, help, error }: {
  label: string; value: T | undefined; onChange: (v: T) => void; options: { value: T; label: string }[]; hint?: ReactNode; help?: string; error?: string
}) {
  const id = useId()
  return (
    <div className="min-w-0">
      <div className="mb-1.5 flex items-center gap-1.5">
        <span id={id} className="text-sm font-medium text-ink">{label}<span className="ml-0.5 text-risk" aria-hidden>*</span></span>
        {help && <InfoTip text={help} label={`About ${label}`} />}
      </div>
      <Segmented label={label} value={value} onChange={onChange} options={options} />
      {error ? <p className="mt-1 text-xs font-medium text-risk" role="alert">{error}</p> : hint ? <p className="mt-1 text-xs text-ink-3">{hint}</p> : null}
    </div>
  )
}

export function FieldGroup({ title, description, icon, children }: { title: string; description?: string; icon?: ReactNode; children: ReactNode }) {
  return (
    <fieldset className="rounded-xl border border-line bg-surface p-5">
      <legend className="sr-only">{title}</legend>
      <div className="mb-4 flex items-start gap-3">
        {icon && <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-navy-50 text-navy-700">{icon}</span>}
        <div>
          <p className="text-[15px] font-semibold text-navy-900" aria-hidden>{title}</p>
          {description && <p className="text-sm text-ink-3">{description}</p>}
        </div>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">{children}</div>
    </fieldset>
  )
}
