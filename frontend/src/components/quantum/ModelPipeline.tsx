import { ChevronDown, ChevronRight } from 'lucide-react'
import { Fragment } from 'react'
import { cx } from '../../lib/format'
import { SERVED_PIPELINE, type PipelineStage } from './pipelineStages'

const kindStyle: Record<PipelineStage['kind'], { box: string; icon: string; tag: string }> = {
  data: { box: 'border-line bg-surface', icon: 'bg-slate-100 text-ink-2', tag: 'Data' },
  classical: { box: 'border-line bg-surface', icon: 'bg-navy-50 text-navy-700', tag: 'Classical' },
  quantum: { box: 'border-teal-600/25 bg-teal-50/60', icon: 'bg-teal-100 text-teal-700', tag: 'Quantum' },
  output: { box: 'border-navy-900 bg-navy-900 text-white', icon: 'bg-white/10 text-teal-400', tag: 'Output' },
}

/**
 * Pipeline flow. Horizontal on wide screens, vertical on narrow ones. `activeIndex`
 * highlights progress only when the caller has real progress to show.
 */
export function ModelPipeline({ stages = SERVED_PIPELINE, orientation = 'auto', compact, activeIndex }: { stages?: PipelineStage[]; orientation?: 'auto' | 'vertical'; compact?: boolean; activeIndex?: number }) {
  const horizontal = orientation === 'auto'
  return (
    <ol
      aria-label="Model pipeline"
      className={cx(
        'flex gap-2',
        horizontal ? 'flex-col lg:flex-row lg:items-stretch' : 'flex-col',
      )}
    >
      {stages.map((s, i) => {
        const st = kindStyle[s.kind]
        const dim = activeIndex !== undefined && i > activeIndex
        return (
          <Fragment key={s.id}>
            <li
              className={cx(
                'relative flex min-w-0 items-center gap-3 rounded-lg border px-3 py-2.5 transition-opacity animate-rise',
                horizontal && 'lg:flex-1 lg:flex-col lg:items-start lg:gap-2 lg:px-3 lg:py-3',
                st.box,
                dim && 'opacity-40',
              )}
              style={{ animationDelay: `${i * 35}ms` }}
            >
              <span className={cx('grid size-8 shrink-0 place-items-center rounded-md', st.icon)}>
                <s.icon className="size-4" aria-hidden />
              </span>
              <span className="min-w-0">
                <span className={cx('block text-[13px] font-semibold leading-tight', s.kind === 'output' ? 'text-white' : 'text-navy-900')}>{s.title}</span>
                {!compact && <span className={cx('mt-0.5 block text-xs leading-snug', s.kind === 'output' ? 'text-slate-300' : 'text-ink-3')}>{s.detail}</span>}
              </span>
              <span className="sr-only">({st.tag} stage)</span>
            </li>
            {i < stages.length - 1 && (
              <li aria-hidden className={cx('flex items-center justify-center text-slate-400', horizontal ? 'h-3 lg:h-auto lg:w-2' : 'h-3')}>
                <ChevronDown className={cx('size-4', horizontal && 'lg:hidden')} />
                {horizontal && <ChevronRight className="hidden size-4 lg:block" />}
              </li>
            )}
          </Fragment>
        )
      })}
    </ol>
  )
}

export function PipelineLegend() {
  return (
    <div className="flex flex-wrap gap-4 text-xs text-ink-3">
      <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm border border-line bg-navy-50" /> Classical stage</span>
      <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm border border-teal-600/30 bg-teal-100" /> Quantum stage</span>
      <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-navy-900" /> Model output</span>
    </div>
  )
}
