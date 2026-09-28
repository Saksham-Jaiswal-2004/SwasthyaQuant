import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { DcqfArm } from '../../data/research'

const BASE = '#2f5ea8'
const HI = '#0d9488'

/** One small multiple: AUC per feature arm for a single label regime. DCQF arms highlighted. */
export function DcqfAblationChart({ arms }: { arms: Record<string, DcqfArm> }) {
  const data = Object.entries(arms).map(([name, a]) => ({ name, auc: a.auc, sd: a.sd, dcqf: name.startsWith('DCQF (') || name === 'raw+DCQF (32)' }))
  return (
    <div className="h-[280px] w-full" aria-hidden>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 50, bottom: 4, left: 4 }}>
          <CartesianGrid horizontal={false} stroke="#e8edf3" />
          <XAxis type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickLine={false} axisLine={false} tick={{ fontSize: 10.5, fill: '#64748b' }} />
          <YAxis type="category" dataKey="name" width={128} tickLine={false} axisLine={false} tick={{ fontSize: 11, fill: '#0f172a' }} />
          <Tooltip
            cursor={{ fill: 'rgb(47 94 168 / 0.06)' }}
            formatter={(v: unknown, _n: unknown, item: { payload?: { sd: number } }) => [`${Number(v).toFixed(3)} ± ${item.payload?.sd.toFixed(3)}`, 'AUC']}
            contentStyle={{ fontSize: 12, borderRadius: 6, borderColor: '#e2e8f0' }}
          />
          <Bar dataKey="auc" radius={[0, 4, 4, 0]} maxBarSize={18}>
            {data.map((d) => <Cell key={d.name} fill={d.dcqf ? HI : BASE} />)}
            <ErrorBar dataKey="sd" width={4} stroke="#0b1f3a" strokeWidth={1} direction="x" />
            <LabelList dataKey="auc" position="right" offset={14} formatter={(v: unknown) => Number(v).toFixed(3)} style={{ fontSize: 10.5, fill: '#0f172a' }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
