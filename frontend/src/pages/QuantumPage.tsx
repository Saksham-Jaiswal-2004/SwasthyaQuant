import { Atom, Cpu, FlaskConical, LineChart, Radio, Waves } from 'lucide-react'
import { DcqfAblationChart } from '../components/charts/DcqfAblationChart'
import { QuantumCircuitVisualization } from '../components/quantum/QuantumCircuitVisualization'
import { Callout, Card, CardHeader, EmptyState, PageHeader, StatusBadge } from '../components/ui/primitives'
import { CLINICAL_REPRESENTATION, DCQF_ABLATION, DCQF_CONFIG, DCQF_FEATURE_SCREENING, DCQF_NULL_TEST, RESEARCH_BENCHMARK, NOISE_SPEC, SHOT_GRID, VQC_CONFIG } from '../data/research'
import { cx } from '../lib/format'

const REGIMES: { key: string; title: string; note: string }[] = [
  { key: 'linear', title: 'Linear labels', note: 'Closest to cardio_train, which is near-linear in blood pressure' },
  { key: 'pairwise', title: 'Pairwise labels', note: 'Built to favour DCQF: cos(xᵢ)cos(xⱼ) structure, no linear signal' },
  { key: 'mixed', title: 'Mixed labels', note: 'Realistic clinical case' },
]

const MODES = [
  { icon: Cpu, name: 'Ideal simulation', status: 'In use', tone: 'good' as const, spec: 'Exact statevector simulation', detail: 'Powers every assessment today.' },
  { icon: Radio, name: 'Finite-shot sampling', status: 'Planned', tone: 'pending' as const, spec: `${SHOT_GRID[0]}–${SHOT_GRID[SHOT_GRID.length - 1].toLocaleString()} shots, ${SHOT_GRID.length} levels`, detail: 'Tests stability under measurement noise.' },
  { icon: Waves, name: 'Noisy simulation', status: 'Planned', tone: 'pending' as const, spec: `Gate error 1q ${NOISE_SPEC.p1q} · 2q ${NOISE_SPEC.p2q} · readout ${NOISE_SPEC.pReadout}`, detail: 'Pessimistic small-device calibration.' },
  { icon: Atom, name: 'IBM Quantum hardware', status: 'Planned', tone: 'pending' as const, spec: 'IBM Quantum Runtime', detail: 'Validation on a real quantum processor.' },
]

export function QuantumPage() {
  const cfg = DCQF_NULL_TEST.config
  return (
    <>
      <PageHeader
        title="Quantum Engine"
        description="The quantum feature layer used by the prototype: 8 clinically selected inputs are encoded into an 8-qubit DCQF circuit, producing 24 observables that can be combined with the classical representation."
      />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ['Selected clinical inputs', `${DCQF_CONFIG.classicalFeatures}`, '8 of the 11 raw attributes after engineering/selection'],
          ['Quantum observables', `${DCQF_CONFIG.quantumFeatures}`, '8 singles + 8 pairs + 8 triples'],
          ['Hybrid vector', `${DCQF_CONFIG.hybridFeatures}`, '8 clinical + 24 DCQF features'],
          ['Quantum parameters', `${DCQF_CONFIG.trainableQuantumParams}`, 'fixed feature extractor; no trainable quantum weights'],
        ].map(([k, v, h]) => (
          <Card key={k} as="div" className="p-4 animate-rise">
            <p className="text-xs font-medium uppercase tracking-wide text-ink-3">{k}</p>
            <p className="mt-2 text-2xl font-semibold text-navy-900 tabular">{v}</p>
            <p className="mt-1 text-xs text-ink-3">{h}</p>
          </Card>
        ))}
      </div>

      <Card className="mt-5">
        <CardHeader title="DCQF quantum feature extractor" subtitle="Fixed 8-qubit digitized counterdiabatic circuit used as a feature transformation layer" icon={<Atom className="size-4" aria-hidden />} />
        <div className="p-5">
          <ol className="mb-4 grid gap-2 text-xs sm:grid-cols-3 lg:grid-cols-6" aria-label="Hybrid QML workflow stages">
            {['8 clinical inputs', 'Angle encoding', `${DCQF_CONFIG.qubits}-qubit register`, '1-step counterdiabatic evolution', 'Z-basis observables', 'Classical ML head'].map((s, i) => (
              <li key={s} className={cx('rounded-md border px-2.5 py-2 font-medium', i >= 1 && i <= 4 ? 'border-teal-600/25 bg-teal-50/60 text-teal-800' : 'border-line bg-surface text-navy-900')}>
                <span className="mr-1 text-ink-3">{i + 1}.</span>{s}
              </li>
            ))}
          </ol>
          <QuantumCircuitVisualization />
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Clinical → quantum mapping" subtitle="The prototype does not send all 11 raw attributes directly to qubits; the fold-safe clinical representation selects 8 inputs." icon={<Atom className="size-4" aria-hidden />} />
        <div className="grid gap-px overflow-hidden bg-line sm:grid-cols-2 lg:grid-cols-4">
          {CLINICAL_REPRESENTATION.map(([z, key, label]) => (
            <div key={z} className="bg-surface p-4">
              <p className="text-xs font-semibold text-teal-700">{z}</p>
              <p className="mt-1 font-semibold text-navy-900">{label}</p>
              <p className="text-xs text-ink-3">{key}</p>
            </div>
          ))}
        </div>
        <div className="border-t border-line p-4 text-xs text-ink-3">
          DCQF layout: Z0–Z7 singles, Z0Z1…Z7Z0 adjacent pairs, and Z0Z1Z2…Z7Z0Z1 adjacent triples on the closed 8-qubit chain.
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Real cardiovascular benchmark" subtitle="Stage B.2 — 70,000 records, 69,959 feature groups, 5 folds × 5 repeats, zero group overlap" icon={<FlaskConical className="size-4" aria-hidden />} />
        <div className="overflow-x-auto p-5">
          <table className="w-full min-w-[700px] text-sm">
            <thead><tr className="border-b border-line text-left text-xs text-ink-3"><th className="py-2 pr-3">Model</th><th>Features</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead>
            <tbody>{RESEARCH_BENCHMARK.models.map((m) => <tr key={m.id} className="border-b border-line/70"><th className="py-2 pr-3 text-left font-semibold text-navy-900">{m.name}</th><td>{m.features}</td><td className="tabular">{m.accuracy.toFixed(4)}</td><td className="tabular">{m.sensitivity.toFixed(4)}</td><td className="tabular">{m.specificity.toFixed(4)}</td><td className="tabular font-semibold">{m.rocAuc.toFixed(4)}</td><td className="tabular">{m.prAuc.toFixed(4)}</td></tr>)}</tbody>
          </table>
          <Callout tone="info" title="What the benchmark actually shows">The classical baseline remains slightly ahead overall. The hybrid model is close, but these experiments do not establish a quantum advantage. This page reports that result rather than overstating it.</Callout>
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Observable-order ablation" subtitle="Stage B.3 — 25 group-aware evaluations using the same classical head" icon={<FlaskConical className="size-4" aria-hidden />} />
        <div className="p-5">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {DCQF_ABLATION.models.slice(0, 4).map((m) => <div key={m.id} className="rounded-lg border border-line p-4"><p className="text-xs text-ink-3">{m.name}</p><p className="mt-1 text-xl font-semibold text-navy-900 tabular">{m.rocAuc.toFixed(4)}</p><p className="text-xs text-ink-3">ROC-AUC</p></div>)}
          </div>
          <div className="mt-4 grid gap-3 md:grid-cols-2"><div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900"><strong>Three-body observables:</strong> standalone triples were the weakest arm (ROC-AUC 0.6322), so they should not be presented as the source of current performance.</div><div className="rounded-md border border-teal-200 bg-teal-50 px-3 py-2 text-xs text-teal-900"><strong>Hybrid result:</strong> adding singles/pairs/triples to the classical vector produced ROC-AUC 0.7882 / 0.7880 / 0.7878 respectively — all below the classical 0.7891.</div></div>
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Training-only DCQF feature screening" subtitle={DCQF_FEATURE_SCREENING.protocol} icon={<LineChart className="size-4" aria-hidden />} />
        <div className="p-5">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {DCQF_FEATURE_SCREENING.topFeatures.map((f) => (
              <div key={f.feature} className="rounded-lg border border-line bg-surface p-3">
                <div className="flex items-center justify-between"><span className="text-xs font-semibold text-teal-700">#{f.rank}</span><span className="font-mono text-sm font-semibold text-navy-900">{f.feature}</span></div>
                <p className="mt-2 text-xs text-ink-3">Train AUC <span className="tabular font-semibold text-ink-2">{f.trainAuc.toFixed(4)}</span> · strength <span className="tabular font-semibold text-ink-2">{f.strength.toFixed(4)}</span></p>
              </div>
            ))}
          </div>
          <Callout tone="info" title="How to interpret this">This ranking was generated from the training split only and was used to screen candidate feature subsets. It is not evidence that these individual observables generalize better than the classical features; the held-out 5×5 benchmark remains the primary comparison.</Callout>
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Execution modes" subtitle="From exact simulation to real quantum hardware" icon={<Cpu className="size-4" aria-hidden />} />
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
        <CardHeader title="Noise degradation benchmark" subtitle="Planned robustness test — not part of the reported cardiovascular benchmark" icon={<LineChart className="size-4" aria-hidden />} />
        <EmptyState icon={<LineChart className="size-5" aria-hidden />} title="Noise benchmark results are not available yet">
          Finite-shot, noisy-simulation and hardware results are not included in the current benchmark. Do not present this prototype as hardware-validated.
        </EmptyState>
      </Card>

      <Card className="mt-5">
        <CardHeader
          title="Quantum feature validation"
          subtitle={`Does the quantum layer add signal beyond classical controls? Controlled test on synthetic data (n = ${cfg.n}, ${cfg.seeds} seeds × ${cfg.folds} folds)`}
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
            <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-[#0d9488]" /> Quantum features (DCQF)</span>
            <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-[#2f5ea8]" /> Raw features or controls</span>
            <span>Whiskers: ± 1 sd across folds</span>
          </div>

          <AblationTable />

          <Callout tone="info" title="What this shows">
            This controlled synthetic test did not show a statistically significant advantage for DCQF over a scrambled null. The product control matched or beat DCQF on the linear and mixed regimes. This is a research control experiment, not a cardiovascular clinical benchmark, and four seeds are not enough to rule out small effects.
          </Callout>
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Hybrid QML models in this application" subtitle="Live stack vs. research-only variations" icon={<Atom className="size-4" aria-hidden />} />
        <dl className="grid gap-px overflow-hidden bg-line sm:grid-cols-3">
          {[
            ['Prototype inference path', 'ClinicalRepresentation → DCQFExtractor → StandardScaler → GradientBoostingClassifier. The backend uses this architecture for patient-level inference. The CV numbers shown above are research benchmarks, not a claim that the deployed artifact reproduces every fold.'],
            ['Research-only hybrid arm', 'A separate fixed random 4-qubit extractor + logistic head exists in the research package. It is not the DCQF model used by the prototype.'],
            ['Other quantum baselines', `VQC (${VQC_CONFIG.qubits} qubits, depth ${VQC_CONFIG.depth}) and the ZZ quantum-kernel SVC are research configurations; neither is the default runtime stack.`],
          ].map(([k, v]) => (
            <div key={k} className="bg-surface p-5">
              <dt className="flex items-center justify-between gap-2 font-semibold text-navy-900">{k} <StatusBadge tone={k === 'Prototype inference path' ? 'good' : 'pending'}>{k === 'Prototype inference path' ? 'In use' : 'Research'}</StatusBadge></dt>
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
