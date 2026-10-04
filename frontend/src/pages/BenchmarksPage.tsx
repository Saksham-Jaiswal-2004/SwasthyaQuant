import { BarChart3, Database, FlaskConical, ShieldCheck } from 'lucide-react'
import { type ReactNode } from 'react'
import { Card, CardHeader, PageHeader } from '../components/ui/primitives'
import { CLINICAL_REPRESENTATION, DCQF_ABLATION, DCQF_SCREENING, MODEL_CATALOG, RESEARCH_BENCHMARK } from '../data/research'

export function BenchmarksPage() {
  return (
    <>
      <PageHeader
        title="Model Performance"
        description="Research benchmarks for the cardiovascular pipeline. Feature comparisons use the same classical Gradient Boosting head so the effect of the representation is isolated."
      />

      <div className="grid gap-3 sm:grid-cols-4">
        <Fact k="Dataset" v="70,000 records" icon={<Database className="size-4" />} />
        <Fact k="Unique groups" v="69,959" icon={<ShieldCheck className="size-4" />} />
        <Fact k="Validation" v="5 × 5 group-aware CV" icon={<FlaskConical className="size-4" />} />
        <Fact k="Train/test group overlap" v="0" icon={<ShieldCheck className="size-4" />} />
      </div>

      <Card className="mt-5">
        <CardHeader title="Primary benchmark — Stage B.2" subtitle="25 held-out evaluations across 5 folds × 5 repeats; preprocessing and feature selection are fit inside each training fold." icon={<BarChart3 className="size-4" />} />
        <div className="overflow-x-auto p-5">
          <table className="w-full min-w-[760px] text-sm">
            <thead><tr className="border-b border-line text-left text-xs text-ink-3">
              <th className="py-2 pr-3">Model</th><th>Features</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th><th>ROC-AUC</th><th>PR-AUC</th>
            </tr></thead>
            <tbody>{RESEARCH_BENCHMARK.models.map((m) => <tr key={m.id} className="border-b border-line/70">
              <th className="py-2 pr-3 text-left font-semibold text-navy-900">{m.name}</th><td className="tabular">{m.features}</td><td className="tabular">{m.accuracy.toFixed(4)}</td><td className="tabular">{m.sensitivity.toFixed(4)}</td><td className="tabular">{m.specificity.toFixed(4)}</td><td className="tabular font-semibold">{m.rocAuc.toFixed(4)}</td><td className="tabular">{m.prAuc.toFixed(4)}</td>
            </tr>)}</tbody>
          </table>
          <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
            <strong>Interpretation:</strong> the classical baseline remains the strongest overall arm in this benchmark. The hybrid representation is close, but does not yet demonstrate a statistically established quantum advantage.
          </div>
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="DCQF ablation — Stage B.3" subtitle="Which quantum observable orders help when added to the same classical representation?" icon={<FlaskConical className="size-4" />} />
        <div className="overflow-x-auto p-5">
          <table className="w-full min-w-[760px] text-sm">
            <thead><tr className="border-b border-line text-left text-xs text-ink-3"><th className="py-2 pr-3">Configuration</th><th>Features</th><th>Accuracy</th><th>Sensitivity</th><th>Specificity</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead>
            <tbody>{DCQF_ABLATION.models.map((m) => <tr key={m.id} className="border-b border-line/70"><th className="py-2 pr-3 text-left font-medium text-navy-900">{m.name}</th><td>{m.features}</td><td className="tabular">{m.accuracy.toFixed(4)}</td><td className="tabular">{m.sensitivity.toFixed(4)}</td><td className="tabular">{m.specificity.toFixed(4)}</td><td className="tabular">{m.rocAuc.toFixed(4)}</td><td className="tabular">{m.prAuc.toFixed(4)}</td></tr>)}</tbody>
          </table>
          <div className="mt-4 grid gap-3 md:grid-cols-2">
            <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-900"><strong>Important:</strong> three-body DCQF features alone performed substantially worse (ROC-AUC 0.6322), so they are not used as a standalone prediction path.</div>
            <div className="rounded-md border border-teal-200 bg-teal-50 px-3 py-2 text-xs text-teal-900"><strong>Finding:</strong> adding singles/pairs/triples to the classical vector did not improve ROC-AUC over the classical baseline in the 25-fold benchmark.</div>
          </div>
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Fast candidate screening — Stage B.4.1" subtitle={DCQF_SCREENING.protocol} icon={<BarChart3 className="size-4" />} />
        <div className="overflow-x-auto p-5">
          <div className="grid gap-3 sm:grid-cols-3">
            <Fact k="Classical baseline" v="Acc 0.7241 · ROC-AUC 0.7922" />
            <Fact k="Best screened accuracy" v="0.7255 (pairs)" />
            <Fact k="Best screened ROC-AUC" v="0.7925 (top-8 / all)" />
          </div>
          <table className="mt-5 w-full min-w-[620px] text-sm">
            <thead><tr className="border-b border-line text-left text-xs text-ink-3"><th className="py-2 pr-3">Configuration</th><th>Features</th><th>Accuracy</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead>
            <tbody>{DCQF_SCREENING.candidates.map((m) => <tr key={m.name} className="border-b border-line/70"><th className="py-2 pr-3 text-left font-medium text-navy-900">{m.name}</th><td>{m.features}</td><td className="tabular">{m.accuracy.toFixed(4)}</td><td className="tabular">{m.rocAuc.toFixed(4)}</td><td className="tabular">{m.prAuc.toFixed(4)}</td></tr>)}</tbody>
          </table>
          <p className="mt-4 text-xs text-ink-3">These screening deltas are small and come from one group-aware split. They are candidate-selection evidence only, not the final benchmark or a claim of quantum advantage.</p>
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Model families implemented in the research stack" subtitle="Inventory only — the primary reported benchmark above isolates the DCQF representation with a common classical head." icon={<ShieldCheck className="size-4" />} />
        <div className="grid gap-px overflow-hidden bg-line sm:grid-cols-2 lg:grid-cols-3">
          {MODEL_CATALOG.map((m) => <div key={m.id} className="bg-surface p-4"><div className="flex items-start justify-between gap-2"><p className="font-semibold text-navy-900">{m.name}</p><span className="rounded-full bg-navy-50 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide text-navy-700">{m.family}</span></div><p className="mt-1 text-xs leading-relaxed text-ink-3">{m.description}</p></div>)}
        </div>
      </Card>

      <Card className="mt-5">
        <CardHeader title="Clinical representation used by DCQF" subtitle="11 raw attributes → engineered representation → 8 selected clinical inputs → 8-qubit encoding" icon={<Database className="size-4" />} />
        <div className="grid gap-px overflow-hidden bg-line sm:grid-cols-2 lg:grid-cols-4">
          {CLINICAL_REPRESENTATION.map(([z, key, label]) => <div key={z} className="bg-surface p-4"><p className="text-xs font-semibold text-teal-700">{z}</p><p className="mt-1 font-semibold text-navy-900">{label}</p><p className="text-xs text-ink-3">{key}</p></div>)}
        </div>
      </Card>

      <p className="mt-5 text-xs text-ink-3">Prototype disclaimer: these are research-model benchmarks, not clinical validation. Swasthya Quant is a decision-support prototype and not a medical diagnostic device.</p>
    </>
  )
}

function Fact({ icon, k, v }: { icon?: ReactNode; k: string; v: string }) {
  return <Card as="div" className="flex items-center gap-3 p-4"><span className="grid size-9 place-items-center rounded-lg bg-navy-50 text-navy-700">{icon ?? <BarChart3 className="size-4" />}</span><div><p className="text-xs text-ink-3">{k}</p><p className="font-semibold text-navy-900">{v}</p></div></Card>
}
