import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { modelName } from '../../data/research'
import { metricDef, type MetricKey, type ModelSummary } from '../../lib/ledger'

const BASE = '#2f5ea8'
const HI = '#0d9488'

interface Datum { id: string; name: string; value: number; sd: number }

function TooltipBody({ active, payload, metric }: { active?: boolean; payload?: { payload: Datum }[]; metric: MetricKey }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const def = metricDef(metric)
  return (
    <div className="rounded-md border border-line bg-surface px-3 py-2 text-xs shadow-lg">
      <p className="font-semibold text-navy-900">{d.name}</p>
      <p className="mt-0.5 text-ink-2 tabular">
        {def.short}: <span className="font-semibold text-ink">{d.value.toFixed(def.unit ? 2 : 3)}{def.unit ?? ''}</span> ± {d.sd.toFixed(def.unit ? 2 : 3)}
      </p>
      <p className="text-ink-3">mean ± sd across folds</p>
    </div>
  )
}

/** Single-series bar chart: one bar per model, best model highlighted. */
export function ModelComparisonChart({ summaries, metric }: { summaries: ModelSummary[]; metric: MetricKey }) {
  const def = metricDef(metric)
  const data: Datum[] = summaries
    .filter((s) => s.mean[metric] !== undefined)
    .map((s) => ({ id: s.model, name: modelName(s.model), value: s.mean[metric]!, sd: s.sd[metric] ?? 0 }))
  const best = data.length ? (def.lowerIsBetter ? Math.min : Math.max)(...data.map((d) => d.value)) : undefined
  const isScore = !def.unit

  return (
    <div className="h-[300px] w-full" aria-hidden>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 24, right: 8, bottom: 4, left: -8 }} barCategoryGap="28%">
          <CartesianGrid vertical={false} stroke="#e8edf3" />
          <XAxis dataKey="name" tickLine={false} axisLine={{ stroke: '#cbd5e1' }} tick={{ fontSize: 11, fill: '#475569' }} interval={0} height={44} tickFormatter={(v: string) => v.replace(' (RBF kernel)', ' RBF')} />
          <YAxis domain={isScore ? [0, 1] : [0, 'auto']} ticks={isScore ? [0, 0.25, 0.5, 0.75, 1] : undefined} tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: '#64748b' }} tickFormatter={(v: number) => (isScore ? v.toFixed(2) : `${v}s`)} width={44} />
          <Tooltip cursor={{ fill: 'rgb(47 94 168 / 0.06)' }} content={<TooltipBody metric={metric} />} />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={56} isAnimationActive animationDuration={500}>
            {data.map((d) => <Cell key={d.id} fill={d.value === best ? HI : BASE} />)}
            <ErrorBar dataKey="sd" width={6} stroke="#0b1f3a" strokeWidth={1} />
            <LabelList dataKey="value" position="top" offset={10} formatter={(v: unknown) => (isScore ? Number(v).toFixed(3) : `${Number(v).toFixed(1)}s`)} style={{ fontSize: 11, fill: '#0f172a', fontWeight: 600 }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
