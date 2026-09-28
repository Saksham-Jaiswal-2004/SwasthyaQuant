export const DASH = '—'

export const fmtScore = (v: number | undefined | null, dp = 3) =>
  v === undefined || v === null || !Number.isFinite(v) ? DASH : v.toFixed(dp)

export const fmtPct = (v: number | undefined | null, dp = 1) =>
  v === undefined || v === null || !Number.isFinite(v) ? DASH : `${(v * 100).toFixed(dp)}%`

export const fmtInt = (v: number | undefined | null) =>
  v === undefined || v === null || !Number.isFinite(v) ? DASH : Math.round(v).toLocaleString('en-IN')

export const fmtNum = (v: number | undefined | null, dp = 1) =>
  v === undefined || v === null || !Number.isFinite(v) ? DASH : v.toFixed(dp)

export const fmtDateTime = (iso: string) =>
  new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })

export const cx = (...parts: (string | false | null | undefined)[]) => parts.filter(Boolean).join(' ')
