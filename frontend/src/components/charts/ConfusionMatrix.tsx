import { fmtInt, fmtPct } from '../../lib/format'

// Sequential single-hue ramp (light -> dark) for row-normalised rates.
const RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95']
const shade = (rate: number) => RAMP[Math.min(RAMP.length - 1, Math.floor(rate * RAMP.length))]
const inkFor = (rate: number) => (rate >= 0.5 ? '#ffffff' : '#0b1f3a')

/**
 * 2×2 confusion matrix. Colour encodes the row-normalised rate (e.g. TP / actual
 * positives = sensitivity), so each row reads as "of these patients, how many…".
 */
export function ConfusionMatrix({ tn, fp, fn, tp, caption }: { tn: number; fp: number; fn: number; tp: number; caption?: string }) {
  const pos = tp + fn
  const neg = tn + fp
  const cells = [
    { row: 'Actual: disease', col: 'Predicted: disease', label: 'True positive', v: tp, rate: pos ? tp / pos : 0, note: 'sensitivity' },
    { row: 'Actual: disease', col: 'Predicted: no disease', label: 'False negative', v: fn, rate: pos ? fn / pos : 0, note: 'missed cases' },
    { row: 'Actual: no disease', col: 'Predicted: disease', label: 'False positive', v: fp, rate: neg ? fp / neg : 0, note: 'false alarms' },
    { row: 'Actual: no disease', col: 'Predicted: no disease', label: 'True negative', v: tn, rate: neg ? tn / neg : 0, note: 'specificity' },
  ]

  return (
    <figure>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-[2px] text-sm">
        <div />
        <div className="px-2 pb-1 text-center text-xs font-medium text-ink-3">Predicted: disease</div>
        <div className="px-2 pb-1 text-center text-xs font-medium text-ink-3">Predicted: no disease</div>
        {[0, 2].map((start) => (
          <div key={start} className="contents">
            <div className="flex items-center pr-2 text-right text-xs font-medium text-ink-3 [writing-mode:horizontal-tb]">
              {cells[start].row.replace('Actual: ', 'Actual ')}
            </div>
            {cells.slice(start, start + 2).map((c) => (
              <div
                key={c.label}
                className="rounded-md px-3 py-4 text-center transition-transform hover:scale-[1.01]"
                style={{ background: shade(c.rate), color: inkFor(c.rate) }}
                title={`${c.label}: ${fmtInt(c.v)} patients (${fmtPct(c.rate)} of ${c.row.toLowerCase()})`}
              >
                <p className="text-[11px] font-semibold uppercase tracking-wide opacity-85">{c.label}</p>
                <p className="mt-1 text-xl font-semibold tabular">{fmtInt(c.v)}</p>
                <p className="text-xs tabular opacity-90">{fmtPct(c.rate)} · {c.note}</p>
              </div>
            ))}
          </div>
        ))}
      </div>
      {caption && <figcaption className="mt-3 text-xs leading-relaxed text-ink-3">{caption}</figcaption>}
    </figure>
  )
}
