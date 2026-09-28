import { DCQF_CONFIG } from '../../data/research'

/**
 * Schematic of the DCQF circuit exactly as implemented in
 * src/qheart/quantum/dcqf.py::_evolve_and_measure (backend config: 8 qubits, pairwise
 * couplings on a closed chain, one Trotter step, Z-string readout of orders 1–3).
 */
export function QuantumCircuitVisualization({ qubits = DCQF_CONFIG.qubits }: { qubits?: number }) {
  const top = 34
  const gap = 36
  const left = 64
  const wireEnd = 800
  const y = (i: number) => top + i * gap
  const height = top + (qubits - 1) * gap + 44

  const col = { h: 100, rz: 172, zz0: 232, zzStep: 22, cd: 440, meas: 560 }
  // Closed-chain pairs (i, i+1 mod n). Each coupling gets its own column so no two read as
  // one multi-qubit gate; they are all diagonal and commute, so the order is immaterial.
  const pairs = Array.from({ length: qubits }, (_, i) => [i, (i + 1) % qubits] as const)
  const zzX = (k: number) => col.zz0 + k * col.zzStep
  const zzMid = zzX((qubits - 1) / 2)

  const gateBox = (x: number, i: number, label: string, fill: string, stroke: string, text = '#0b1f3a', w = 34) => (
    <g key={`${label}-${x}-${i}`}>
      <rect x={x - w / 2} y={y(i) - 12} width={w} height={24} rx={5} fill={fill} stroke={stroke} />
      <text x={x} y={y(i) + 4} textAnchor="middle" fontSize={11} fontWeight={600} fill={text} fontFamily="JetBrains Mono, monospace">{label}</text>
    </g>
  )

  const zz = (x: number, a: number, b: number, key: string) => {
    const [lo, hi] = a < b ? [a, b] : [b, a]
    return (
      <g key={key}>
        <line x1={x} x2={x} y1={y(lo)} y2={y(hi)} stroke="#0d9488" strokeWidth={2} />
        <circle cx={x} cy={y(lo)} r={5} fill="#0d9488" />
        <circle cx={x} cy={y(hi)} r={5} fill="#0d9488" />
      </g>
    )
  }

  return (
    <figure className="w-full">
      <div className="overflow-x-auto rounded-lg border border-line bg-[#fbfcfe]">
        <svg viewBox={`0 0 900 ${height}`} className="h-auto w-full min-w-[640px]" role="img" aria-labelledby="dcqf-title dcqf-desc">
          <title id="dcqf-title">DCQF quantum feature-extraction circuit</title>
          <desc id="dcqf-desc">
            {`${qubits} qubits, one per selected clinical feature. Each qubit starts in the plus state via a Hadamard, receives a Z-field rotation from its clinical angle, pairwise ZZ couplings weighted by mutual information along a closed chain, then a counterdiabatic kick of Y and Y-Z terms, and is measured in the Z basis. The readout is 24 expectation values: 8 single-qubit, 8 neighbouring pairs and 8 neighbouring triples.`}
          </desc>

          {/* stage labels */}
          {[
            { x: col.h, t: 'Prep |+⟩' },
            { x: col.rz, t: 'Fields' },
            { x: zzMid, t: 'MI couplings (ZZ)' },
            { x: col.cd, t: 'CD kick (AGP)' },
            { x: col.meas, t: 'Measure Z' },
          ].map((s) => (
            <text key={s.t} x={s.x} y={14} textAnchor="middle" fontSize={10.5} fontWeight={600} fill="#64748b" letterSpacing="0.06em">{s.t.toUpperCase()}</text>
          ))}

          {Array.from({ length: qubits }, (_, i) => (
            <g key={`w${i}`}>
              <text x={8} y={y(i) + 4} fontSize={11} fill="#475569" fontFamily="JetBrains Mono, monospace">q{i}</text>
              <text x={34} y={y(i) + 4} fontSize={11} fill="#94a3b8" fontFamily="JetBrains Mono, monospace">|0⟩</text>
              <line x1={left} x2={wireEnd - 180} y1={y(i)} y2={y(i)} stroke="#cbd5e1" strokeWidth={1.5} />
              {gateBox(col.h, i, 'H', '#ffffff', '#94a3b8')}
              {gateBox(col.rz, i, `Rz(x${i})`, '#e3eaf5', '#2f5ea8', '#0b1f3a', 56)}
            </g>
          ))}

          {pairs.map(([a, b], k) => zz(zzX(k), a, b, `zz${k}`))}
          <text x={zzX(qubits - 1)} y={y(qubits - 1) + 28} textAnchor="middle" fontSize={10} fill="#0f766e">wrap q{qubits - 1}–q0</text>

          {/* counterdiabatic block spans all wires */}
          <rect x={col.cd - 42} y={y(0) - 16} width={84} height={y(qubits - 1) - y(0) + 32} rx={8} fill="#f0fdfa" stroke="#0d9488" strokeDasharray="4 3" />
          <text x={col.cd} y={y(Math.floor((qubits - 1) / 2)) + 4} textAnchor="middle" fontSize={12} fontWeight={700} fill="#0f766e" fontFamily="JetBrains Mono, monospace">e^(−iA·dt)</text>
          <text x={col.cd} y={y(Math.floor((qubits - 1) / 2)) + 22} textAnchor="middle" fontSize={10} fill="#0f766e">Y + Y·Z terms</text>

          {Array.from({ length: qubits }, (_, i) => (
            <g key={`m${i}`}>
              <rect x={col.meas - 15} y={y(i) - 12} width={30} height={24} rx={5} fill="#0b1f3a" />
              <path d={`M${col.meas - 9} ${y(i) + 5} A 9 9 0 0 1 ${col.meas + 9} ${y(i) + 5}`} fill="none" stroke="#2dd4bf" strokeWidth={1.6} />
              <line x1={col.meas} y1={y(i) + 5} x2={col.meas + 6} y2={y(i) - 5} stroke="#2dd4bf" strokeWidth={1.6} />
            </g>
          ))}

          {/* readout groups */}
          {[
            { t: `${qubits} × ⟨Zᵢ⟩`, s: 'one-body' },
            { t: `${qubits} × ⟨ZᵢZᵢ₊₁⟩`, s: 'two-body chain' },
            { t: `${qubits} × ⟨ZᵢZᵢ₊₁Zᵢ₊₂⟩`, s: 'three-body chain' },
          ].map((g, k) => {
            const cy = top + ((qubits - 1) * gap) * (0.18 + k * 0.32)
            return (
              <g key={g.s}>
                <rect x={640} y={cy - 22} width={240} height={44} rx={8} fill="#ffffff" stroke="#cbd5e1" />
                <text x={656} y={cy - 3} fontSize={12} fontWeight={600} fill="#0b1f3a" fontFamily="JetBrains Mono, monospace">{g.t}</text>
                <text x={656} y={cy + 13} fontSize={10.5} fill="#64748b">{g.s} expectation values</text>
              </g>
            )
          })}
          <line x1={588} x2={636} y1={top + ((qubits - 1) * gap) / 2} y2={top + ((qubits - 1) * gap) / 2} stroke="#94a3b8" strokeWidth={1.5} markerEnd="url(#arrow)" />
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0 0 L10 5 L0 10 z" fill="#94a3b8" />
            </marker>
          </defs>
        </svg>
      </div>
      <figcaption className="mt-2 text-xs leading-relaxed text-ink-3">
        Schematic of <span className="font-mono">DCQFExtractor</span> as configured in the backend: {qubits} qubits (one per selected clinical angle), pairwise couplings
        weighted by mutual information measured on the training fold, a single Trotter step (dt = 1), exact statevector expectations, and {DCQF_CONFIG.trainableQuantumParams} trainable quantum parameters.
        Output: {DCQF_CONFIG.quantumFeatures} quantum features.
      </figcaption>
    </figure>
  )
}
