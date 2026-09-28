import { Atom, BarChart3, Brain, Database, Filter, Layers, ShieldCheck } from 'lucide-react'
import type { ReactNode } from 'react'
import { Card, PageHeader } from '../components/ui/primitives'
import { DATASET } from '../data/research'

export function MethodologyPage() {
  return (
    <>
      <PageHeader
        eyebrow="Methodology"
        title="How Swasthya Quant works"
        description="A leakage-safe, seeded and reproducible pipeline in which classical and quantum models are scored by the same harness under the same rules."
      />
      <div className="grid gap-5 lg:grid-cols-2">
        <Section icon={<Database className="size-4" />} title="1 · Data">
          <p>{DATASET.name} (<span className="font-mono">{DATASET.id}</span>): {DATASET.records.toLocaleString('en-IN')} patient records, {DATASET.rawFeatures} features and a binary <span className="font-mono">cardio</span> target.</p>
          <ul>
            <li>Numeric: age, height, weight, systolic and diastolic blood pressure</li>
            <li>Binary: smoking, alcohol intake, physical activity</li>
            <li>Categorical: gender, cholesterol level, glucose level</li>
          </ul>
        </Section>
        <Section icon={<Filter className="size-4" />} title="2 · Preprocessing">
          <ul>
            <li><b>Cleaning:</b> physiologically impossible values become missing, never clipped. Diastolic above systolic is invalidated.</li>
            <li><b>Derived features:</b> age in years, BMI, pulse pressure, mean arterial pressure.</li>
            <li><b>Imputation and scaling:</b> median imputation and scaling fitted inside each training fold only.</li>
            <li><b>Feature selection:</b> MI-mRMR picks 8 of 12 candidates (redundancy penalty 0.5), per fold. PCA-8 is an alternative configuration.</li>
            <li><b>Angle scaling:</b> fixed clinical bounds to [0, π], chosen before seeing data, so the quantum encoding has no fitted state.</li>
          </ul>
        </Section>
        <Section icon={<BarChart3 className="size-4" />} title="3 · Classical baselines">
          <ul>
            <li>Logistic regression: the linear floor</li>
            <li>SVM with RBF kernel: the direct counterpart to the quantum kernel</li>
            <li>Random forest (400 trees) and XGBoost (400 trees, depth 3)</li>
            <li>MLP (32-16): an unmatched neural baseline of about 1,200 parameters</li>
          </ul>
        </Section>
        <Section icon={<Atom className="size-4" />} title="4 · Quantum layer">
          <ul>
            <li><b>DCQF:</b> clinical angles set the longitudinal fields, and mutual information between features sets the couplings. One counterdiabatic Trotter step follows, then 1-, 2- and 3-body Z readouts give 24 features. No trainable quantum parameters.</li>
            <li><b>VQC:</b> 4 qubits, dense angle encoding, depth 6, 96 trainable angles.</li>
            <li><b>Quantum kernel:</b> ZZ fidelity kernel feeding an SVC.</li>
          </ul>
        </Section>
        <Section icon={<Layers className="size-4" />} title="5 · Hybrid layer">
          <p>8 classical angles plus 24 quantum expectation values give 32 hybrid features. These pass through a fitted StandardScaler to a Gradient Boosting classifier, and the positive-class probability is the risk estimate.</p>
          <p>A parameter-matched classical control (Control-C) is defined so that any quantum gain is measured against equal model capacity.</p>
        </Section>
        <Section icon={<Brain className="size-4" />} title="6 · Evaluation">
          <ul>
            <li>5 repeats × stratified 5-fold CV (25 fits per model), reported as mean ± sd. No single-split numbers.</li>
            <li>Decision threshold chosen on training folds by Youden&rsquo;s J.</li>
            <li>Headline metrics: sensitivity and specificity. Primary: PR-AUC. Also ROC-AUC, F1, PPV, NPV, MCC, Brier, ECE, accuracy.</li>
            <li>Paired tests: McNemar, Nadeau–Bengio corrected t-test, Holm–Bonferroni correction.</li>
            <li>Computational cost: fit time per fold, circuit-evaluation budgets, parameter counts.</li>
          </ul>
        </Section>
      </div>

      <Card className="mt-5 p-5">
        <div className="flex items-start gap-3">
          <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-teal-50 text-teal-700"><ShieldCheck className="size-4" aria-hidden /></span>
          <div className="text-sm leading-relaxed text-ink-2">
            <p className="font-semibold text-navy-900">What the project does not claim</p>
            <p className="mt-1">No quantum advantage in accuracy. No clinical utility: this is not a medical device. No generalisation to new hospitals. No hardware advantage: a simulator result is never labelled as hardware. These limits are set out in <span className="font-mono">docs/RESULTS_PROTOCOL.md</span>, which was written before any model was fitted.</p>
          </div>
        </div>
      </Card>
    </>
  )
}

function Section({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center gap-3">
        <span className="grid size-8 place-items-center rounded-lg bg-navy-50 text-navy-700" aria-hidden>{icon}</span>
        <h2 className="text-[15px] font-semibold text-navy-900">{title}</h2>
      </div>
      <div className="space-y-2 text-sm leading-relaxed text-ink-2 [&_li]:relative [&_li]:pl-4 [&_li]:before:absolute [&_li]:before:left-0 [&_li]:before:top-[9px] [&_li]:before:size-1.5 [&_li]:before:rounded-full [&_li]:before:bg-teal-500 [&_ul]:space-y-1.5 [&_b]:font-semibold [&_b]:text-ink">
        {children}
      </div>
    </Card>
  )
}
