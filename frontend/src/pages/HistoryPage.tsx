import { ClipboardPlus, History, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { PredictionResult } from '../components/clinical/PredictionResult'
import { Button, Card, EmptyState, PageHeader, StatusBadge } from '../components/ui/primitives'
import { useAppState } from '../hooks/appStateContext'
import { GENDER_LABELS } from '../lib/clinical'
import { cx, fmtDateTime } from '../lib/format'

export function HistoryPage() {
  const { history, removeRecord, clearHistory } = useAppState()
  const [openId, setOpenId] = useState<string | null>(history[0]?.id ?? null)
  const open = history.find((h) => h.id === openId)

  return (
    <>
      <PageHeader
        eyebrow="Prediction History"
        title="Assessment history"
        description="Assessments run in this browser. The backend exposes no history endpoint, so nothing is stored server-side."
        actions={history.length > 0 && (
          <Button variant="secondary" size="sm" onClick={() => { if (confirm('Clear all saved assessments from this browser?')) clearHistory() }}>
            <Trash2 className="size-4" aria-hidden /> Clear history
          </Button>
        )}
      />

      {history.length === 0 ? (
        <Card>
          <EmptyState icon={<History className="size-5" aria-hidden />} title="No assessments yet"
            action={<Link to="/app/assess" className="inline-flex h-10 items-center gap-2 rounded-lg bg-navy-900 px-4 text-sm font-medium text-white hover:bg-navy-800"><ClipboardPlus className="size-4" aria-hidden /> Start Disease Assessment</Link>}>
            Results returned by the model will be listed here.
          </EmptyState>
        </Card>
      ) : (
        <div className="grid gap-5 lg:grid-cols-[340px_minmax(0,1fr)]">
          <Card as="div" className="self-start">
            <ul className="divide-y divide-line" aria-label="Saved assessments">
              {history.map((r) => (
                <li key={r.id} className="flex items-stretch">
                  <button
                    onClick={() => setOpenId(r.id)}
                    aria-current={r.id === openId}
                    className={cx('min-w-0 flex-1 px-4 py-3 text-left text-sm hover:bg-navy-50/60', r.id === openId && 'bg-navy-50')}
                  >
                    <span className="flex items-center justify-between gap-2">
                      <span className="font-semibold text-navy-900 tabular">{r.result.risk_percentage}%</span>
                      <span className="flex gap-1">
                        {r.source === 'demo' && <StatusBadge tone="warn">Demo</StatusBadge>}
                        <StatusBadge tone={r.result.prediction ? 'risk' : 'good'}>{r.result.prediction ? 'Higher' : 'Lower'}</StatusBadge>
                      </span>
                    </span>
                    <span className="mt-0.5 block text-xs text-ink-3">{GENDER_LABELS[r.input.gender]}, {r.input.age} y · {r.input.ap_hi}/{r.input.ap_lo} mmHg</span>
                    <span className="block text-xs text-ink-3">{fmtDateTime(r.createdAt)}</span>
                  </button>
                  <button onClick={() => removeRecord(r.id)} className="px-3 text-ink-3 hover:text-risk" aria-label={`Delete assessment from ${fmtDateTime(r.createdAt)}`}>
                    <Trash2 className="size-4" aria-hidden />
                  </button>
                </li>
              ))}
            </ul>
          </Card>
          <div>{open ? <PredictionResult key={open.id} record={open} /> : <Card><EmptyState title="Select an assessment" /></Card>}</div>
        </div>
      )}
    </>
  )
}
