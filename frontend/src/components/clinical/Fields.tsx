import { useId, type ReactNode } from 'react'
import { cx } from '../../lib/format'
import { InfoTip, Segmented } from '../ui/primitives'

function Label({ htmlFor, id, label, required, help }: { htmlFor?: string; id?: string; label: string; required?: boolean; help?: string }) {
  const Tag = htmlFor ? 'label' : 'span'
  return (
    <div className="mb-1.5 flex items-center gap-1.5">
      <Tag htmlFor={htmlFor} id={id} className="text-[13px] font-medium text-ink-2">
        {label}
        {required && <span className="sr-only"> (required)</span>}
      </Tag>
      {help && <InfoTip text={help} label={`About ${label}`} />}
    </div>
  )
}

export function NumberField({ label, value, onChange, unit, min, max, step = 1, hint, error, help, required = true }: {
  label: string; value: number | ''; onChange: (v: number | '') => void; unit: string; min: number; max: number; step?: number; hint?: ReactNode; error?: string; help?: string; required?: boolean
}) {
  const id = useId()
  const describedBy = error ? `${id}-err` : hint ? `${id}-hint` : undefined
  return (
    <div className="min-w-0">
      <Label htmlFor={id} label={label} required={required} help={help} />
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
          title={`${min}–${max} ${unit}`}
          aria-invalid={!!error}
          aria-describedby={describedBy}
          onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
          className="h-full w-full min-w-0 rounded-l-lg bg-transparent px-3 text-[15px] text-ink tabular outline-none"
        />
        <span className="shrink-0 pr-3 text-xs font-medium text-ink-3">{unit}</span>
      </div>
      {error ? (
        <p id={`${id}-err`} className="mt-1 text-xs font-medium text-risk" role="alert">{error}</p>
      ) : hint ? (
        <p id={`${id}-hint`} className="mt-1 text-xs text-ink-3">{hint}</p>
      ) : null}
    </div>
  )
}

export function ChoiceField<T extends number>({ label, value, onChange, options, hint, help, error }: {
  label: string; value: T | undefined; onChange: (v: T) => void; options: { value: T; label: string }[]; hint?: ReactNode; help?: string; error?: string
}) {
  const id = useId()
  return (
    <div className="min-w-0">
      <Label id={id} label={label} help={help} />
      <Segmented label={label} value={value} onChange={onChange} options={options} fullWidth />
      {error ? <p className="mt-1 text-xs font-medium text-risk" role="alert">{error}</p> : hint ? <p className="mt-1 text-xs text-ink-3">{hint}</p> : null}
    </div>
  )
}

/** Binary yes/no input rendered as an accessible switch row. */
export function SwitchField({ label, description, checked, onChange }: { label: string; description?: string; checked: boolean; onChange: (v: boolean) => void }) {
  const id = useId()
  return (
    <div className="flex items-center justify-between gap-3 py-2">
      <span className="min-w-0">
        <span id={id} className="block text-sm font-medium text-ink">{label}</span>
        {description && <span className="block text-xs text-ink-3">{description}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-labelledby={id}
        onClick={() => onChange(!checked)}
        className={cx('relative h-6 w-11 shrink-0 rounded-full transition-colors', checked ? 'bg-teal-600' : 'bg-slate-300')}
      >
        <span className={cx('absolute top-0.5 left-0.5 size-5 rounded-full bg-white shadow transition-transform', checked && 'translate-x-5')} />
      </button>
    </div>
  )
}

export function PanelSection({ title, icon, children, className }: { title: string; icon?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <fieldset className={cx('border-b border-line px-5 py-4 last:border-b-0', className)}>
      <legend className="float-left mb-3 flex w-full items-center gap-2 text-xs font-semibold uppercase tracking-[0.1em] text-navy-700">
        {icon}
        {title}
      </legend>
      <div className="clear-both space-y-3.5">{children}</div>
    </fieldset>
  )
}
