import { RefreshCw } from 'lucide-react'
import { useAppState } from '../../hooks/appStateContext'
import { cx } from '../../lib/format'

/** Live status from GET /api/health. */
export function ServiceStatusPill() {
  const { service, refreshHealth } = useAppState()
  const { dot, text, detail } =
    service.state === 'checking' ? { dot: 'bg-slate-400 animate-pulse', text: 'Checking service…', detail: '' }
    : service.state === 'offline' ? { dot: 'bg-risk', text: 'API offline', detail: 'The inference API could not be reached.' }
    : service.health.model_loaded ? { dot: 'bg-good', text: 'Model ready', detail: 'API online, model artifact loaded.' }
    : { dot: 'bg-warn', text: 'Model not loaded', detail: 'API online; no trained model artifact is loaded (health: degraded).' }

  return (
    <button
      type="button"
      onClick={refreshHealth}
      title={`${detail} Click to re-check.`}
      className="group inline-flex h-8 items-center gap-2 rounded-full border border-line bg-surface px-3 text-xs font-medium text-ink-2 hover:bg-navy-50"
    >
      <span className={cx('size-2 rounded-full', dot)} aria-hidden />
      <span>{text}</span>
      <RefreshCw className="size-3 text-ink-3 opacity-0 transition-opacity group-hover:opacity-100" aria-hidden />
      <span className="sr-only">. {detail} Activate to re-check.</span>
    </button>
  )
}
