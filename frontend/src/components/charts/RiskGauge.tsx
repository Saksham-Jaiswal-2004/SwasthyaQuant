import { useEffect, useState } from 'react'

/**
 * Semicircular gauge for a model probability. The 0.5 tick marks the backend's decision
 * threshold (prediction_service.py: pred = prob >= 0.5). Colour follows the backend's
 * predicted class, not a made-up risk band.
 */
export function RiskGauge({ probability, predictedPositive, demo }: { probability: number; predictedPositive: boolean; demo?: boolean }) {
  const [shown, setShown] = useState(0)
  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(probability))
    return () => cancelAnimationFrame(id)
  }, [probability])

  const r = 80
  const circ = Math.PI * r
  const color = demo ? '#94a3b8' : predictedPositive ? '#b91c1c' : '#15803d'
  const pct = (probability * 100).toFixed(1)

  return (
    <div className="relative mx-auto w-full max-w-[260px]">
      <svg viewBox="0 0 200 120" className="w-full" role="img" aria-label={`Model-estimated risk probability ${pct} percent`}>
        <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#e8edf3" strokeWidth={14} strokeLinecap="round" />
        <path
          d="M20 100 A80 80 0 0 1 180 100"
          fill="none"
          stroke={color}
          strokeWidth={14}
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={circ * (1 - shown)}
          style={{ transition: 'stroke-dashoffset 700ms cubic-bezier(0.2,0.7,0.2,1)' }}
        />
        {/* decision threshold at 0.5 */}
        <line x1="100" y1="12" x2="100" y2="28" stroke="#0b1f3a" strokeWidth={2} />
        <text x="100" y="9" textAnchor="middle" fontSize="8" fill="#64748b">0.5 threshold</text>
        <text x="20" y="116" textAnchor="middle" fontSize="9" fill="#94a3b8">0%</text>
        <text x="180" y="116" textAnchor="middle" fontSize="9" fill="#94a3b8">100%</text>
      </svg>
      <div className="pointer-events-none absolute inset-x-0 bottom-3 text-center">
        <p className="text-4xl font-semibold tracking-tight text-navy-900 tabular">{pct}<span className="text-2xl text-ink-3">%</span></p>
        <p className="text-xs font-medium uppercase tracking-wide text-ink-3">Model-predicted risk</p>
      </div>
    </div>
  )
}
