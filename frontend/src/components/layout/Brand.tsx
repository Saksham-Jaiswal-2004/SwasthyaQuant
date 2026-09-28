import { cx } from '../../lib/format'

export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={cx('shrink-0', className)} aria-hidden>
      <rect width="32" height="32" rx="8" fill="#0b1f3a" />
      <rect x="0.5" y="0.5" width="31" height="31" rx="7.5" fill="none" stroke="#2dd4bf" strokeOpacity="0.25" />
      <path d="M5 17h5l2.5-6 4 11 3-8 1.5 3H27" fill="none" stroke="#2dd4bf" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function Brand({ inverted, compact }: { inverted?: boolean; compact?: boolean }) {
  return (
    <span className="flex items-center gap-2.5">
      <BrandMark className="size-8" />
      {!compact && (
        <span className="leading-tight">
          <span className={cx('block text-[15px] font-semibold tracking-tight', inverted ? 'text-white' : 'text-navy-900')}>
            Swasthya <span className={inverted ? 'text-teal-400' : 'text-teal-600'}>Quant</span>
          </span>
          <span className={cx('block text-[10.5px] font-medium uppercase tracking-[0.14em]', inverted ? 'text-slate-400' : 'text-ink-3')}>
            Cardiac Risk AI
          </span>
        </span>
      )}
    </span>
  )
}
