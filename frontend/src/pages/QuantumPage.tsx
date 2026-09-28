import { Atom, Cpu, FlaskConical, LineChart, Radio, Waves } from 'lucide-react'
import { DcqfAblationChart } from '../components/charts/DcqfAblationChart'
import { QuantumCircuitVisualization } from '../components/quantum/QuantumCircuitVisualization'
import { Callout, Card, CardHeader, EmptyState, PageHeader, SourceNote, StatusBadge } from '../components/ui/primitives'
import { DCQF_CONFIG, DCQF_NULL_TEST, NOISE_SPEC, SHOT_GRID, VQC_CONFIG } from '../data/research'
import { cx } from '../lib/format'

const REGIMES: { key: string; title: string; note: string }[] = [
  { key: 'linear', title: 'Linear labels', note: 'Closest to cardio_train, which is near-linear in blood pressure' },
  { key: 'pairwise', title: 'Pairwise labels', note: 'Built to favour DCQF: cos(xᵢ)cos(xⱼ) structure, no linear signal' },
  { key: 'mixed', title: 'Mixed labels', note: 'Realistic clinical case' },
]

const MODES = [
  { icon: Cpu, name: 'Ideal simulation', status: 'In use', tone: 'good' as const, spec: 'Exact statevector (shots = None)', detail: 'Configuration served by the backend DCQF extractor.' },
  { icon: Radio, name: 'Finite-shot sampling', status: 'Configured', tone: 'info' as const, spec: `Shot grid ${SHOT_GRID[0]}–${SHOT_GRID[SHOT_GRID.length - 1].toLocaleString()} (${SHOT_GRID.length} log-spaced points)`, detail: 'degradation_curve() in quantum/noise.py' },
  { icon: Waves, name: 'Synthetic noise (Aer)', status: 'Configured', tone: 'info' as const, spec: `Depolarising 1q ${NOISE_SPEC.p1q} · 2q ${NOISE_SPEC.p2q} · readout ${NOISE_SPEC.pReadout}`, detail: `NoiseSpec "${NOISE_SPEC.label}"` },
  { icon: Atom, name: 'IBM Quantum hardware', status: 'Not run', tone: 'pending' as const, spec: 'qiskit-ibm-runtime sampler', detail: 'runtime_sampler() in quantum/noise.py; needs IBM_QUANTUM_TOKEN' },
]

export function QuantumPage() {
  const cfg = DCQF_NULL_TEST.config
  return (
    <>
      <PageHeader
        eyebrow="Quantum Hardware"
        title="Quantum layer and hardware readiness"
        description="The circuit the backend runs, the execution modes the codebase defines for testing robustness to noise, and the ablation that tests whether the quantum step contributes."
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ['Qubits', `${DCQF_CONFIG.qubits}`, 'one per selected clinical feature'],
          ['Quantum features', `${DCQF_CONFIG.quantumFeatures}`, 'Z strings of order 1, 2, 3'],
          ['Trotter steps', `${DCQF_CONFIG.trotterSteps}`, 'impulse regime, dt = 1'],
          ['Trainable quantum params', `${DCQF_CONFIG.trainableQuantumParams}`, 'couplings from mutual information'],
        ].map(([k, v, h]) => (
          <Card key={k} as="div" className="p-4 animate-rise">
            <p className="text-xs font-medium uppercase tracking-wide text-ink-3">{k}</p>
            <p className="mt-2 text-2xl font-semibold text-navy-900 tabular">{v}</p>
            <p className="mt-1 text-xs text-ink-3">{h}</p>
          </Card>
        ))}
      </div>

      <Card className="mt-5">
        <CardHeader title="Quantum workflow" subtitle="Digitized counterdiabatic quantum feature extraction (DCQF)" icon={<Atom className="size-4" aria-hidden />} />
        <div className="p-5">
          <ol className="mb-4 grid gap-2 text-xs sm:grid-cols-3 lg:grid-cols-6" aria-label="Quantum workflow stages">
            {['Classical input (8 angles)', 'Feature encoding', `${DCQF_CONFIG.qubits}-qubit register`, 'Counterdiabatic circuit', 'Z-basis measurement', 'Classical classifier'].map((s, i) => (
              <li key={s} className={cx('rounded-md border px-2.5 py-2 font-medium', i >= 1 && i <= 4 ? 'border-teal-600/25 bg-teal-50/60 text-teal-800' : 'border-line bg-surface text-navy-900')}>
                <span className="mr-1 text-ink-3">{i + 1}.</span>{s}
              </li>
            ))}
          </ol>
          <QuantumCircuitVisualization />
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Execution modes" subtitle="Simulator → noisy simulator → hardware, in that order (quantum/noise.py)" icon={<Cpu className="size-4" aria-hidden />} />
        <div className="grid gap-px overflow-hidden bg-line sm:grid-cols-2 xl:grid-cols-4">
          {MODES.map((m) => (
            <div key={m.name} className="bg-surface p-5">
              <div className="flex items-center justify-between gap-2">
                <m.icon className="size-5 text-navy-700" aria-hidden />
                <StatusBadge tone={m.tone}>{m.status}</StatusBadge>
              </div>
              <p className="mt-3 font-semibold text-navy-900">{m.name}</p>
              <p className="mt-1 text-sm text-ink-2">{m.spec}</p>
              <p className="mt-1 text-xs text-ink-3">{m.detail}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Noise degradation benchmark" subtitle="Metric against shot count and two-qubit error rate" icon={<LineChart className="size-4" aria-hidden />} />
        <EmptyState icon={<LineChart className="size-5" aria-hidden />} title="Noise benchmark results are not available yet">
          Ideal, noisy-simulator and hardware runs have not been written to <span className="font-mono">results/runs/</span>.
          Accuracy, recall, circuit depth and execution time for each mode will appear here once they have.
        </EmptyState>
      </Card>

      <Card className="mt-5">
        <CardHeader
          title="Does the quantum step contribute? DCQF ablation"
          subtitle={`AUC per feature set on synthetic data · n = ${cfg.n}, ${cfg.seeds} seeds × ${cfg.folds} folds, one shared logistic head`}
          icon={<FlaskConical className="size-4" aria-hidden />}
        />
        <div className="space-y-5 p-5">
          <div className="grid gap-5 lg:grid-cols-3">
            {REGIMES.map((r) => (
              <figure key={r.key} className="min-w-0">
                <figcaption className="mb-1">
                  <p className="text-sm font-semibold text-navy-900">{r.title}</p>
                  <p className="text-xs text-ink-3">{r.note}</p>
                </figcaption>
                <DcqfAblationChart arms={DCQF_NULL_TEST.results[r.key]} />
              </figure>
            ))}
          </div>
          <div className="flex flex-wrap gap-4 text-xs text-ink-3">
            <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-[#0d9488]" /> DCQF arm</span>
            <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-[#2f5ea8]" /> Raw features or controls</span>
            <span>Whiskers: ± 1 sd across folds</span>
          </div>

          <AblationTable />

          <Callout tone="info" title="What this shows">
            DCQF did not beat its scrambled null (couplings permuted off their variables) in any regime, and the classical chain-product
            control matched or beat it. Its apparent gain over raw features comes from expanding 8 columns into 24. The platform reports
            this openly. The test is limited: synthetic data, 4 seeds, and small effects would go undetected (docs/DCQF_FINDINGS.md).
          </Callout>
          <SourceNote>results/runs/dcqf_null_test.json</SourceNote>
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Other quantum research arms" subtitle="Defined in configs/model; no ledger rows yet" icon={<Atom className="size-4" aria-hidden />} />
        <dl className="grid gap-px overflow-hidden bg-line sm:grid-cols-3">
          {[
            ['Variational Quantum Classifier', `${VQC_CONFIG.qubits} qubits · depth ${VQC_CONFIG.depth} · ${VQC_CONFIG.trainableAngles} angles + ${VQC_CONFIG.bias} bias · ${VQC_CONFIG.entangler} entangler. Parameter-shift cost: ${VQC_CONFIG.paramShiftEvalsPerStep} circuit evaluations per step per sample.`],
            ['Quantum kernel (ZZ) + SVC', '4 qubits · 2 reps · linear entanglement. O(N²) circuit evaluations for the Gram matrix.'],
            ['Quantum extractor + head', 'Fixed random 4-qubit circuit (depth 6) feeding a logistic head; compared against a parameter-matched classical projection.'],
          ].map(([k, v]) => (
            <div key={k} className="bg-surface p-5">
              <dt className="flex items-center justify-between gap-2 font-semibold text-navy-900">{k} <StatusBadge tone="pending">Awaiting run</StatusBadge></dt>
              <dd className="mt-1 text-sm text-ink-2">{v}</dd>
            </div>
          ))}
        </dl>
      </Card>
    </>
  )
}

function AblationTable() {
  const rows = [
    { label: 'DCQF vs scrambled null', key: 'DCQF scrambled (24)' },
    { label: 'DCQF vs chain products', key: 'chain products (24)' },
    { label: 'DCQF vs raw features', key: 'raw X (8)' },
  ]
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[620px] text-sm">
        <caption className="sr-only">DCQF AUC differences against each comparator, per label regime</caption>
        <thead>
          <tr className="border-b border-line text-left text-xs text-ink-3">
            <th scope="col" className="py-2 pr-3 font-medium">Comparison (DCQF − comparator)</th>
            {REGIMES.map((r) => <th key={r.key} scope="col" className="px-3 py-2 text-right font-medium">{r.title}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} className="border-b border-line/70">
              <th scope="row" className="py-2 pr-3 text-left font-medium text-navy-900">{row.label}</th>
              {REGIMES.map((r) => {
                const res = DCQF_NULL_TEST.results[r.key]
                const diff = res['DCQF (24)'].auc - res[row.key].auc
                const sig = res[row.key].significant
                return (
                  <td key={r.key} className="px-3 py-2 text-right tabular">
                    <span className="font-semibold text-ink">{diff >= 0 ? '+' : '−'}{Math.abs(diff).toFixed(4)}</span>
                    <span className="ml-2">{sig ? <StatusBadge tone="neutral">significant</StatusBadge> : <StatusBadge tone="pending">not significant</StatusBadge>}</span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
