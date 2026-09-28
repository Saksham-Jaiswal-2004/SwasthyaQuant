import { AlertTriangle, CheckCircle2, CircleDashed, Info, Loader2, RefreshCw, ShieldAlert, XCircle } from 'lucide-react'
import { useId, useState, type ButtonHTMLAttributes, type ReactNode } from 'react'
import { cx } from '../../lib/format'

export function Card({ className, children, as: As = 'section' }: { className?: string; children: ReactNode; as?: 'section' | 'div' | 'article' }) {
  return <As className={cx('rounded-xl border border-line bg-surface shadow-[0_1px_2px_rgb(15_23_42/0.04)]', className)}>{children}</As>
}

export function CardHeader({ title, subtitle, action, icon }: { title: ReactNode; subtitle?: ReactNode; action?: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-line px-5 py-4">
      <div className="flex min-w-0 items-start gap-3">
        {icon && <div className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-lg bg-navy-50 text-navy-700">{icon}</div>}
        <div className="min-w-0">
          <h2 className="text-[15px] font-semibold text-ink">{title}</h2>
          {subtitle && <p className="mt-0.5 text-sm text-ink-3">{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  )
}

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4 animate-rise">
      <div className="max-w-3xl">
        {eyebrow && <p className="mb-1 text-xs font-semibold uppercase tracking-[0.12em] text-teal-700">{eyebrow}</p>}
        <h1 className="text-2xl font-semibold tracking-tight text-navy-900 sm:text-[28px]">{title}</h1>
        {description && <p className="mt-2 text-[15px] leading-relaxed text-ink-2">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </header>
  )
}

export function SectionHeader({ title, description }: { title: string; description?: ReactNode }) {
  return (
    <div className="mb-3 mt-8 first:mt-0">
      <h2 className="text-base font-semibold text-navy-900">{title}</h2>
      {description && <p className="mt-0.5 text-sm text-ink-3">{description}</p>}
    </div>
  )
}

type Tone = 'good' | 'warn' | 'risk' | 'neutral' | 'info' | 'pending'
const toneClass: Record<Tone, string> = {
  good: 'bg-good-bg text-good ring-good/20',
  warn: 'bg-warn-bg text-warn ring-warn/25',
  risk: 'bg-risk-bg text-risk ring-risk/20',
  neutral: 'bg-slate-100 text-ink-2 ring-slate-300/60',
  info: 'bg-teal-50 text-teal-700 ring-teal-600/20',
  pending: 'bg-slate-50 text-ink-3 ring-slate-300/60',
}
const toneIcon: Record<Tone, ReactNode> = {
  good: <CheckCircle2 className="size-3.5" aria-hidden />,
  warn: <AlertTriangle className="size-3.5" aria-hidden />,
  risk: <XCircle className="size-3.5" aria-hidden />,
  neutral: null,
  info: <Info className="size-3.5" aria-hidden />,
  pending: <CircleDashed className="size-3.5" aria-hidden />,
}

export function StatusBadge({ tone, children, className }: { tone: Tone; children: ReactNode; className?: string }) {
  return (
    <span className={cx('inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset', toneClass[tone], className)}>
      {toneIcon[tone]}
      {children}
    </span>
  )
}

export function StatCard({ label, value, hint, badge, icon, pending }: { label: string; value: ReactNode; hint?: ReactNode; badge?: ReactNode; icon?: ReactNode; pending?: boolean }) {
  return (
    <Card as="div" className="p-4 animate-rise">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-medium uppercase tracking-wide text-ink-3">{label}</p>
        {icon && <span className="text-ink-3">{icon}</span>}
      </div>
      <p className={cx('mt-2 text-xl font-semibold tabular', pending ? 'text-ink-3' : 'text-navy-900')}>{value}</p>
      {(hint || badge) && (
        <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs text-ink-3">
          {badge}
          {hint && <span>{hint}</span>}
        </div>
      )}
    </Card>
  )
}

export function Button({ variant = 'primary', size = 'md', className, children, ...rest }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'ghost' | 'teal'; size?: 'sm' | 'md' | 'lg' }) {
  const v = {
    primary: 'bg-navy-900 text-white hover:bg-navy-800 disabled:bg-navy-700/70',
    teal: 'bg-teal-600 text-white hover:bg-teal-700',
    secondary: 'bg-surface text-navy-900 ring-1 ring-inset ring-line hover:bg-navy-50',
    ghost: 'text-ink-2 hover:bg-slate-100',
  }[variant]
  const s = { sm: 'h-8 px-3 text-sm', md: 'h-10 px-4 text-sm', lg: 'h-12 px-6 text-[15px]' }[size]
  return (
    <button className={cx('inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors disabled:cursor-not-allowed', v, s, className)} {...rest}>
      {children}
    </button>
  )
}

export function LoadingState({ label = 'Loading…', className }: { label?: string; className?: string }) {
  return (
    <div role="status" aria-live="polite" className={cx('flex items-center justify-center gap-3 py-12 text-sm text-ink-3', className)}>
      <Loader2 className="size-5 animate-spin text-teal-600" aria-hidden />
      {label}
    </div>
  )
}

export function EmptyState({ icon, title, children, action, className }: { icon?: ReactNode; title: string; children?: ReactNode; action?: ReactNode; className?: string }) {
  return (
    <div className={cx('flex flex-col items-center justify-center px-6 py-10 text-center', className)}>
      <div className="mb-3 grid size-11 place-items-center rounded-full bg-slate-100 text-ink-3">{icon ?? <CircleDashed className="size-5" aria-hidden />}</div>
      <p className="font-medium text-ink">{title}</p>
      {children && <div className="mt-1 max-w-md text-sm leading-relaxed text-ink-3">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function ErrorState({ title = 'Something went wrong', children, onRetry, className }: { title?: string; children?: ReactNode; onRetry?: () => void; className?: string }) {
  return (
    <div role="alert" className={cx('flex flex-col items-center justify-center px-6 py-10 text-center', className)}>
      <div className="mb-3 grid size-11 place-items-center rounded-full bg-risk-bg text-risk"><AlertTriangle className="size-5" aria-hidden /></div>
      <p className="font-medium text-ink">{title}</p>
      {children && <div className="mt-1 max-w-md text-sm leading-relaxed text-ink-3">{children}</div>}
      {onRetry && (
        <Button variant="secondary" size="sm" className="mt-4" onClick={onRetry}>
          <RefreshCw className="size-4" aria-hidden /> Try again
        </Button>
      )}
    </div>
  )
}

export function Disclaimer({ children, className }: { children?: ReactNode; className?: string }) {
  return (
    <p className={cx('flex items-start gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs leading-relaxed text-ink-3 ring-1 ring-inset ring-line', className)}>
      <ShieldAlert className="mt-px size-3.5 shrink-0" aria-hidden />
      <span>{children ?? 'This result is an AI-generated risk estimate and is not a medical diagnosis. It has not been validated for clinical use.'}</span>
    </p>
  )
}

export function Callout({ tone = 'info', title, children, className }: { tone?: 'info' | 'warn' | 'risk' | 'good'; title?: ReactNode; children: ReactNode; className?: string }) {
  const c = {
    info: 'border-teal-600/20 bg-teal-50/70 text-teal-900',
    warn: 'border-warn/25 bg-warn-bg text-amber-900',
    risk: 'border-risk/20 bg-risk-bg text-red-900',
    good: 'border-good/20 bg-good-bg text-green-900',
  }[tone]
  const I = { info: Info, warn: AlertTriangle, risk: XCircle, good: CheckCircle2 }[tone]
  return (
    <div className={cx('flex gap-3 rounded-lg border px-4 py-3 text-sm leading-relaxed', c, className)}>
      <I className="mt-0.5 size-4 shrink-0" aria-hidden />
      <div className="min-w-0">
        {title && <p className="font-semibold">{title}</p>}
        <div className={title ? 'mt-0.5 opacity-90' : ''}>{children}</div>
      </div>
    </div>
  )
}

/** Accessible hover/focus tooltip for short help text. */
export function InfoTip({ text, label = 'More information' }: { text: string; label?: string }) {
  const id = useId()
  const [open, setOpen] = useState(false)
  return (
    <span className="relative inline-flex align-middle">
      <button
        type="button"
        aria-label={label}
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((o) => !o)}
        className="grid size-4 place-items-center rounded-full text-ink-3 hover:text-navy-700"
      >
        <Info className="size-3.5" aria-hidden />
      </button>
      {open && (
        <span id={id} role="tooltip" className="absolute bottom-full left-1/2 z-30 mb-2 w-60 -translate-x-1/2 rounded-md bg-navy-900 px-3 py-2 text-xs font-normal normal-case leading-relaxed tracking-normal text-white shadow-lg animate-fade">
          {text}
        </span>
      )}
    </span>
  )
}

export function SourceNote({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cx('text-xs text-ink-3', className)}>Source: <span className="font-mono text-[11px]">{children}</span></p>
}

export function Segmented<T extends string | number>({ value, onChange, options, label, size = 'md' }: { value: T | undefined; onChange: (v: T) => void; options: { value: T; label: string }[]; label: string; size?: 'sm' | 'md' }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex w-full flex-wrap gap-1 rounded-lg bg-slate-100 p-1 sm:w-auto">
      {options.map((o) => {
        const active = o.value === value
        return (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(o.value)}
            className={cx(
              'flex-1 whitespace-nowrap rounded-md font-medium transition-colors sm:flex-none',
              size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3 py-1.5 text-sm',
              active ? 'bg-surface text-navy-900 shadow-sm ring-1 ring-line' : 'text-ink-2 hover:text-navy-900',
            )}
          >
            {o.label}
          </button>
        )
      })}
    </div>
  )
}
