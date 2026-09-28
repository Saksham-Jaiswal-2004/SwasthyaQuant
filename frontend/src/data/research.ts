// Facts read from the repository. Every value here cites the file it comes from so the UI
// never shows a number that cannot be traced. Nothing in this file is invented.

import ledgerCsv from '../../../results/runs/ledger.csv?raw'
import dcqfNullTest from '../../../results/runs/dcqf_null_test.json'

/** Build-time snapshot of the append-only research ledger. */
export const LEDGER_SNAPSHOT = { text: ledgerCsv, path: 'results/runs/ledger.csv' }

export interface DcqfArm { auc: number; sd: number; delta?: number; t_fold?: number; t_seed?: number; significant?: boolean }
export interface DcqfNullTest {
  config: { seeds: number; n: number; folds: number }
  seconds: number
  results: Record<string, Record<string, DcqfArm>>
}
/** Synthetic-data ablation from scripts/dcqf_null_test.py (see docs/DCQF_FINDINGS.md). */
export const DCQF_NULL_TEST = dcqfNullTest as DcqfNullTest

export interface ModelInfo {
  id: string
  name: string
  family: 'classical' | 'quantum' | 'hybrid' | 'control'
  description: string
  config: string
}

/** Model arms defined in configs/model/*.yaml and src/qheart/models/registry.py. */
export const MODEL_CATALOG: ModelInfo[] = [
  { id: 'logreg', name: 'Logistic Regression', family: 'classical', description: 'The honest linear floor (balanced class weights).', config: 'configs/model/logreg.yaml' },
  { id: 'svm_rbf', name: 'SVM (RBF kernel)', family: 'classical', description: 'Direct classical counterpart to the quantum kernel: same SVC, only the kernel differs.', config: 'configs/model/svm_rbf.yaml' },
  { id: 'rf', name: 'Random Forest', family: 'classical', description: '400 trees, balanced class weights.', config: 'configs/model/rf.yaml' },
  { id: 'xgboost', name: 'XGBoost', family: 'classical', description: '400 trees, max depth 3. Expected to win the accuracy column.', config: 'configs/model/xgboost.yaml' },
  { id: 'mlp', name: 'MLP (32-16)', family: 'classical', description: 'Unmatched neural baseline, ~1,200 parameters.', config: 'configs/model/mlp.yaml' },
  { id: 'hybrid_dcqf_gb', name: 'Hybrid DCQF + Gradient Boosting', family: 'hybrid', description: '8 clinical angles + 24 DCQF quantum features → StandardScaler → Gradient Boosting. The model served by /api/predict.', config: 'backend/README.md' },
  { id: 'hybrid_extractor', name: 'Quantum extractor + head', family: 'hybrid', description: 'Fixed random 4-qubit circuit feeding a logistic head (HQF-CC pattern).', config: 'configs/model/hybrid_extractor.yaml' },
  { id: 'vqc_dense', name: 'Variational Quantum Classifier', family: 'quantum', description: '4 qubits, dense angle encoding, depth 6 → 96 trainable angles + 1 bias.', config: 'configs/model/vqc_dense.yaml' },
  { id: 'qkernel_zz', name: 'Quantum kernel (ZZ) + SVC', family: 'quantum', description: 'ZZ fidelity kernel, 4 qubits, 2 reps. O(N²) circuit evaluations.', config: 'configs/model/qkernel_zz.yaml' },
  { id: 'control_c', name: 'Parameter-matched control', family: 'control', description: 'Classical control sized from the live circuit (95 / 108 params brackets).', config: 'configs/model/control_c.yaml' },
]

export const modelName = (id: string) => MODEL_CATALOG.find((m) => m.id === id)?.name ?? id

/** The model served by the backend (backend/README.md, prediction_service.py). */
export const SERVED_MODEL = {
  name: 'Hybrid DCQF + Gradient Boosting',
  short: 'Hybrid Quantum-Classical',
  classifier: 'GradientBoostingClassifier',
  threshold: 0.5,
}

/** DCQF extractor configuration used by the backend (backend/app/services/preprocessing_service.py). */
export const DCQF_CONFIG = {
  qubits: 8, // one per selected clinical feature (ClinicalRepresentation k=8)
  encodedOrders: [2],
  readoutOrders: [1, 2, 3],
  quantumFeatures: 24, // 8 one-body + 8 two-body + 8 three-body Z strings on a closed chain
  classicalFeatures: 8,
  hybridFeatures: 32,
  trotterSteps: 1,
  dt: 1.0,
  shots: null as number | null, // exact statevector expectations
  trainableQuantumParams: 0,
  miBins: 4,
  source: 'backend/app/services/preprocessing_service.py',
}

/** VQC research arm (configs/model/vqc_dense.yaml, configs/base.yaml). */
export const VQC_CONFIG = { qubits: 4, depth: 6, trainableAngles: 96, bias: 1, encoding: 'dense angle', entangler: 'ring', paramShiftEvalsPerStep: 194 }

/** Noise model and shot grid (src/qheart/quantum/noise.py). */
export const NOISE_SPEC = { p1q: 0.001, p2q: 0.01, pReadout: 0.02, label: 'small-device-pessimistic' }
export const SHOT_GRID = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

/** Candidate clinical features and rationale (src/qheart/features/select.py). */
export const FEATURE_RATIONALE: Record<string, { label: string; text: string }> = {
  age_years: { label: 'Age', text: 'Continuous age representation; captures age-related cardiovascular risk.' },
  bmi: { label: 'BMI', text: 'Body-mass index; captures weight relative to height.' },
  ap_hi: { label: 'Systolic BP', text: 'Systolic blood pressure; strong cardiovascular risk signal.' },
  ap_lo: { label: 'Diastolic BP', text: 'Diastolic blood pressure; complementary blood-pressure information.' },
  pulse_pressure: { label: 'Pulse pressure', text: 'Systolic minus diastolic pressure; captures pressure amplitude.' },
  map: { label: 'Mean arterial pressure', text: 'Compact representation of overall arterial pressure.' },
  cholesterol: { label: 'Cholesterol', text: 'Ordinal cholesterol category with substantial target-rate separation.' },
  gluc: { label: 'Glucose', text: 'Ordinal glucose category with measurable target-rate separation.' },
  active: { label: 'Physical activity', text: 'Physical-activity indicator representing a behavioral dimension.' },
  gender: { label: 'Gender', text: 'Demographic variable retained as an initial candidate for subgroup analysis.' },
  smoke: { label: 'Smoking', text: 'Smoking indicator; included as a candidate despite weak marginal association.' },
  alco: { label: 'Alcohol', text: 'Alcohol-use indicator; included as a candidate despite weak marginal association.' },
}

export const DATASET = {
  name: 'Cardiovascular Disease dataset',
  id: 'cardio_70000',
  records: 70000,
  rawFeatures: 11,
  target: 'cardio',
  source: 'configs/base.yaml · data/raw/cardio_train.csv',
}

/** Research reference actually cited in the repository (src/qheart/quantum/dcqf.py). */
export const REFERENCES = [
  {
    authors: 'Simen, Flores-Garrigós, De Oliveira, Alvarado Barrios, Gomez Cadavid, Dalal, Solano, Hegade & Zhang',
    title: 'Digitized counterdiabatic quantum feature extraction',
    venue: 'Scientific Reports (2026), accepted article in press',
    doi: '10.1038/s41598-026-67564-0',
    note: 'Implemented as DCQFExtractor. Replicated with scrambled and product controls; see docs/DCQF_FINDINGS.md.',
  },
]
