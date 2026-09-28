import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

/** Shape a future attribution endpoint (e.g. SHAP from src/qheart/explain/shap_classical.py) should return. */
export interface FeatureAttribution {
  feature: string
  /** Signed contribution: positive raises predicted risk, negative lowers it. */
  value: number
}

// Diverging poles: blue (lowers risk) <-> red (raises risk), neutral zero line.
const UP = '#c2413b'
const DOWN = '#2f5ea8'

/** Diverging horizontal bar chart of signed attributions, sorted by magnitude. */
export function FeatureImportanceChart({ data, height }: { data: FeatureAttribution[]; height?: number }) {
  const sorted = [...data].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  const h = height ?? Math.max(180, sorted.length * 34 + 40)
  return (
    <div style={{ height: h }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={sorted} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid horizontal={false} stroke="#e8edf3" />
          <XAxis type="number" tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: '#64748b' }} />
          <YAxis type="category" dataKey="feature" width={130} tickLine={false} axisLine={false} tick={{ fontSize: 12, fill: '#0f172a' }} />
          <ReferenceLine x={0} stroke="#94a3b8" />
          <Tooltip
            cursor={{ fill: 'rgb(47 94 168 / 0.06)' }}
            formatter={(v: unknown) => [Number(v).toFixed(4), 'Contribution']}
            contentStyle={{ fontSize: 12, borderRadius: 6, borderColor: '#e2e8f0' }}
          />
          <Bar dataKey="value" radius={4} maxBarSize={20}>
            {sorted.map((d) => <Cell key={d.feature} fill={d.value >= 0 ? UP : DOWN} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
