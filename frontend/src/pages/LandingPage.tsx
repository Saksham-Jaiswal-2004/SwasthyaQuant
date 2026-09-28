import { ArrowRight, Atom, BarChart3, HeartPulse, Lightbulb, ShieldCheck, Waves } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Brand } from '../components/layout/Brand'
import { ServiceStatusPill } from '../components/layout/ServiceStatusPill'
import { ModelPipeline, PipelineLegend } from '../components/quantum/ModelPipeline'
import { PS_ID, PS_TITLE, TAGLINE } from '../config'
import { DATASET, DCQF_CONFIG } from '../data/research'

const HIGHLIGHTS = [
  { icon: Atom, title: 'Hybrid Quantum ML', text: `An ${DCQF_CONFIG.qubits}-qubit counterdiabatic feature map adds ${DCQF_CONFIG.quantumFeatures} quantum features to the clinical inputs before a classical classifier.` },
  { icon: HeartPulse, title: 'Clinical Risk Prediction', text: `Cardiovascular risk estimated from ${DATASET.rawFeatures} routine measurements, including blood pressure, BMI and lab markers.` },
  { icon: Lightbulb, title: 'Explainable AI', text: 'Named clinical features with stated rationale. SHAP for the classifier and parameter-shift saliency for the circuit, kept separate.' },
  { icon: Waves, title: 'Quantum Noise Benchmarking', text: 'Ideal, finite-shot and depolarising-noise execution modes to test hardware readiness before a real device.' },
]

export function LandingPage() {
  return (
    <div className="min-h-dvh bg-canvas">
      <section className="relative overflow-hidden bg-navy-900 text-white">
        <div aria-hidden className="pointer-events-none absolute inset-0 opacity-[0.07] [background-image:linear-gradient(#fff_1px,transparent_1px),linear-gradient(90deg,#fff_1px,transparent_1px)] [background-size:40px_40px]" />
        <div aria-hidden className="pointer-events-none absolute -right-40 -top-40 size-[520px] rounded-full bg-teal-500/10 blur-3xl" />
        <div className="relative mx-auto max-w-6xl px-4 sm:px-6">
          <nav className="flex h-16 items-center justify-between" aria-label="Site">
            <Brand inverted />
            <div className="flex items-center gap-2">
              <span className="hidden sm:block"><ServiceStatusPill /></span>
              <Link to="/app" className="rounded-lg px-3 py-2 text-sm font-medium text-slate-200 hover:bg-white/10">Open dashboard</Link>
            </div>
          </nav>

          <div className="grid gap-10 py-16 lg:grid-cols-[1.15fr_1fr] lg:items-center lg:py-24">
            <div className="animate-rise">
              <p className="inline-flex items-center gap-2 rounded-full border border-teal-400/30 bg-teal-400/10 px-3 py-1 text-xs font-medium text-teal-300">
                SIH 2026 · PS {PS_ID}
              </p>
              <h1 className="mt-5 text-4xl font-semibold tracking-tight sm:text-5xl">Swasthya Quant</h1>
              <p className="mt-3 text-xl font-medium text-teal-300 sm:text-2xl">{TAGLINE}</p>
              <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-slate-300">
                Clinical prediction powered by hybrid quantum-classical machine learning. Every model is scored under one leakage-safe
                protocol and benchmarked against the classical controls that could explain its result.
              </p>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link to="/app/assess" className="inline-flex h-12 items-center gap-2 rounded-lg bg-teal-500 px-6 text-[15px] font-semibold text-navy-950 hover:bg-teal-400">
                  Start Disease Assessment <ArrowRight className="size-4" aria-hidden />
                </Link>
                <Link to="/app/benchmarks" className="inline-flex h-12 items-center gap-2 rounded-lg px-6 text-[15px] font-medium text-white ring-1 ring-inset ring-white/25 hover:bg-white/10">
                  <BarChart3 className="size-4" aria-hidden /> View Model Benchmarks
                </Link>
              </div>
            </div>

            <div className="animate-rise rounded-2xl border border-white/10 bg-white/[0.04] p-5 backdrop-blur [animation-delay:120ms]">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">At a glance</p>
              <dl className="mt-4 grid grid-cols-2 gap-4">
                {[
                  ['Patient records', DATASET.records.toLocaleString('en-IN')],
                  ['Clinical inputs', String(DATASET.rawFeatures)],
                  ['Qubits in feature map', String(DCQF_CONFIG.qubits)],
                  ['Hybrid features', `${DCQF_CONFIG.classicalFeatures} + ${DCQF_CONFIG.quantumFeatures}`],
                ].map(([k, v]) => (
                  <div key={k} className="rounded-lg bg-white/[0.04] p-3">
                    <dt className="text-xs text-slate-400">{k}</dt>
                    <dd className="mt-1 text-2xl font-semibold tabular">{v}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-4 text-xs leading-relaxed text-slate-400">Evaluation: 5 repeats × stratified 5-fold cross-validation, with thresholds chosen on training folds only.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6" aria-labelledby="tech-h">
        <h2 id="tech-h" className="text-2xl font-semibold tracking-tight text-navy-900">Technology</h2>
        <p className="mt-1 text-ink-2">Four capabilities, one reproducible research harness.</p>
        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {HIGHLIGHTS.map((h, i) => (
            <article key={h.title} className="rounded-xl border border-line bg-surface p-5 animate-rise" style={{ animationDelay: `${i * 60}ms` }}>
              <span className="grid size-10 place-items-center rounded-lg bg-teal-50 text-teal-700"><h.icon className="size-5" aria-hidden /></span>
              <h3 className="mt-4 font-semibold text-navy-900">{h.title}</h3>
              <p className="mt-1.5 text-sm leading-relaxed text-ink-2">{h.text}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="border-y border-line bg-surface" aria-labelledby="pipe-h">
        <div className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
          <h2 id="pipe-h" className="text-2xl font-semibold tracking-tight text-navy-900">From clinical data to explained risk</h2>
          <p className="mt-1 text-ink-2">The inference pipeline the platform runs for each patient.</p>
          <div className="mt-8"><ModelPipeline /></div>
          <div className="mt-4"><PipelineLegend /></div>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
        <div className="grid gap-6 rounded-2xl bg-navy-900 p-8 text-white md:grid-cols-[auto_1fr_auto] md:items-center">
          <ShieldCheck className="size-10 text-teal-400" aria-hidden />
          <div>
            <p className="text-lg font-semibold">Built to detect weak results, including our own</p>
            <p className="mt-1 text-sm leading-relaxed text-slate-300">Quantum components are tested against scrambled and dimension-matched classical controls. Results are reported whether or not they favour the quantum model.</p>
          </div>
          <Link to="/app/quantum" className="inline-flex h-11 items-center gap-2 whitespace-nowrap rounded-lg bg-white px-5 text-sm font-semibold text-navy-900 hover:bg-slate-100">
            See the evidence <ArrowRight className="size-4" aria-hidden />
          </Link>
        </div>
      </section>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-6 text-xs text-ink-3 sm:px-6">
          <span>Swasthya Quant · SIH PS {PS_ID}: {PS_TITLE}</span>
          <span>Research prototype. Not a medical device. Outputs are model-estimated risk, not a diagnosis.</span>
        </div>
      </footer>
    </div>
  )
}
