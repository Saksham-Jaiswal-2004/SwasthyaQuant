import { Atom, Binary, Brain, ClipboardList, Filter, Gauge, Layers, Scale, Sigma, type LucideIcon } from 'lucide-react'

export interface PipelineStage {
  id: string
  title: string
  detail: string
  kind: 'data' | 'classical' | 'quantum' | 'output'
  icon: LucideIcon
}

/** The inference chain served by /api/predict (backend/README.md, preprocessing_service.py). */
export const SERVED_PIPELINE: PipelineStage[] = [
  { id: 'input', title: 'Clinical input', detail: '11 raw features', kind: 'data', icon: ClipboardList },
  { id: 'derive', title: 'Clinical features', detail: 'BMI, pulse pressure, MAP', kind: 'classical', icon: Sigma },
  { id: 'select', title: 'Feature selection', detail: 'MI-mRMR, k = 8', kind: 'classical', icon: Filter },
  { id: 'encode', title: 'Angle encoding', detail: '8 angles in [0, π]', kind: 'quantum', icon: Binary },
  { id: 'dcqf', title: 'DCQF circuit', detail: '8 qubits · 1 Trotter step', kind: 'quantum', icon: Atom },
  { id: 'hybrid', title: 'Hybrid features', detail: '8 classical + 24 quantum', kind: 'quantum', icon: Layers },
  { id: 'scale', title: 'Standardise', detail: 'Fitted scaler (z-score)', kind: 'classical', icon: Scale },
  { id: 'clf', title: 'Hybrid classifier', detail: 'Gradient Boosting', kind: 'classical', icon: Brain },
  { id: 'risk', title: 'Risk estimate', detail: 'P(cardio = 1)', kind: 'output', icon: Gauge },
]

/** Condensed five-step view of SERVED_PIPELINE for narrow, product-facing contexts. */
export const SUMMARY_PIPELINE: PipelineStage[] = [
  { id: 'input', title: 'Health metrics', detail: '11 routine inputs', kind: 'data', icon: ClipboardList },
  { id: 'features', title: 'Clinical features', detail: 'Derived and selected (8)', kind: 'classical', icon: Filter },
  { id: 'quantum', title: 'Quantum encoding', detail: '8-qubit DCQF circuit', kind: 'quantum', icon: Atom },
  { id: 'clf', title: 'Hybrid classifier', detail: '32 features, gradient boosting', kind: 'classical', icon: Brain },
  { id: 'risk', title: 'Risk estimate', detail: 'Probability of disease', kind: 'output', icon: Gauge },
]
