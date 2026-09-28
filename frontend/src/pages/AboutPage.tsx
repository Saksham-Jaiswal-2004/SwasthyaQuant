import { BookMarked, Code2, ExternalLink } from 'lucide-react'
import { Brand } from '../components/layout/Brand'
import { Card, CardHeader, PageHeader } from '../components/ui/primitives'
import { PS_ID, PS_TITLE, TAGLINE } from '../config'
import { REFERENCES } from '../data/research'

// Only technologies present in the repository's requirements / package.json.
const STACK: { group: string; items: string[] }[] = [
  { group: 'Frontend', items: ['React 19', 'TypeScript', 'Vite', 'Tailwind CSS', 'Recharts', 'React Router'] },
  { group: 'API', items: ['Python', 'FastAPI', 'Pydantic', 'Uvicorn'] },
  { group: 'Machine learning', items: ['scikit-learn', 'XGBoost', 'NumPy', 'pandas', 'SciPy'] },
  { group: 'Quantum', items: ['PennyLane', 'Qiskit', 'Qiskit Aer', 'Qiskit Machine Learning', 'IBM Quantum Runtime', 'PyTorch'] },
  { group: 'Explainability', items: ['SHAP', 'Parameter-shift saliency'] },
]

export function AboutPage() {
  return (
    <>
      <PageHeader eyebrow="About" title="About the project" />
      <Card className="overflow-hidden">
        <div className="bg-navy-900 px-6 py-8 text-white">
          <Brand inverted />
          <p className="mt-5 text-xl font-semibold tracking-tight">{TAGLINE}</p>
          <p className="mt-1 text-sm text-slate-300">Smart India Hackathon · Problem Statement {PS_ID}</p>
          <p className="text-sm text-slate-300">{PS_TITLE}</p>
        </div>
        <div className="grid gap-6 p-6 text-sm leading-relaxed text-ink-2 md:grid-cols-2">
          <p>
            Swasthya Quant is a research platform for early cardiovascular disease risk detection. It combines a quantum feature map with
            classical machine learning, and it benchmarks every quantum component against classical controls under one fold-safe protocol.
          </p>
          <p>
            Its focus is rigour, not accuracy supremacy. Every figure traces to an append-only results ledger. Quantum results are reported
            beside the null models that could explain them away. Every prediction is presented as a model-estimated risk, never as a diagnosis.
          </p>
        </div>
      </Card>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader title="Technology stack" subtitle="As declared in the repository" icon={<Code2 className="size-4" aria-hidden />} />
          <dl className="space-y-4 p-5">
            {STACK.map((s) => (
              <div key={s.group}>
                <dt className="text-xs font-medium uppercase tracking-wide text-ink-3">{s.group}</dt>
                <dd className="mt-1.5 flex flex-wrap gap-1.5">
                  {s.items.map((i) => <span key={i} className="rounded-md bg-navy-50 px-2 py-1 text-xs font-medium text-navy-800">{i}</span>)}
                </dd>
              </div>
            ))}
          </dl>
        </Card>
        <Card>
          <CardHeader title="Research inspiration" subtitle="Works cited in the codebase" icon={<BookMarked className="size-4" aria-hidden />} />
          <ul className="space-y-4 p-5 text-sm">
            {REFERENCES.map((r) => (
              <li key={r.doi}>
                <p className="font-semibold text-navy-900">{r.title}</p>
                <p className="text-ink-2">{r.authors}</p>
                <p className="text-ink-3">{r.venue}</p>
                <a href={`https://doi.org/${r.doi}`} target="_blank" rel="noreferrer" className="mt-1 inline-flex items-center gap-1 font-mono text-xs text-teal-700 hover:underline">
                  doi:{r.doi} <ExternalLink className="size-3" aria-hidden />
                </a>
                <p className="mt-1 text-xs text-ink-3">{r.note}</p>
              </li>
            ))}
          </ul>
        </Card>
      </div>
    </>
  )
}
