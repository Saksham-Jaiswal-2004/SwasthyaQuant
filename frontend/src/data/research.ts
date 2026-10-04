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
  { id: 'xgboost', name: 'XGBoost', family: 'classical', description: '400 trees, max depth 3. Strong classical nonlinear baseline used for comparison.', config: 'configs/model/xgboost.yaml' },
  { id: 'mlp', name: 'MLP (32-16)', family: 'classical', description: 'Unmatched neural baseline, ~1,200 parameters.', config: 'configs/model/mlp.yaml' },
  { id: 'hybrid_dcqf_gb', name: 'Hybrid DCQF + Gradient Boosting', family: 'hybrid', description: '8 clinical angles + 24 DCQF quantum features → StandardScaler → Gradient Boosting. The model served by /api/predict.', config: 'backend/README.md' },
  { id: 'hybrid_all', name: 'Hybrid all-features model', family: 'hybrid', description: 'Uses all 32 hybrid features (8 clinical + 24 DCQF) with a GradientBoostingClassifier as modeled in scripts/stage_b2_hybrid_dcqf.py.', config: 'scripts/stage_b2_hybrid_dcqf.py' },
  { id: 'hybrid_shap', name: 'Hybrid SHAP-selected model', family: 'hybrid', description: 'Uses the SHAP-ranked subset of the hybrid feature vector, selected from the full 32-feature DCQF stack in the B.2 evaluation pass.', config: 'scripts/stage_b2_hybrid_dcqf.py' },
  { id: 'hybrid_extractor', name: 'Quantum extractor + head', family: 'hybrid', description: 'Fixed random 4-qubit circuit feeding a logistic head (HQF-CC pattern).', config: 'configs/model/hybrid_extractor.yaml' },
  { id: 'vqc_dense', name: 'Variational Quantum Classifier', family: 'quantum', description: '4 qubits, dense angle encoding, depth 6 → 96 trainable angles + 1 bias.', config: 'configs/model/vqc_dense.yaml' },
  { id: 'qkernel_zz', name: 'Quantum kernel (ZZ) + SVC', family: 'quantum', description: 'ZZ fidelity kernel, 4 qubits, 2 reps. O(N²) circuit evaluations.', config: 'configs/model/qkernel_zz.yaml' },
  { id: 'control_c', name: 'Parameter-matched control', family: 'control', description: 'Classical control sized from the live circuit (95 / 108 params brackets).', config: 'configs/model/control_c.yaml' },
]

export const modelName = (id: string) => MODEL_CATALOG.find((m) => m.id === id)?.name ?? id

export const RESEARCH_BENCHMARK = {
  protocol: 'Stage B.2 — 5-fold × 5-repeat group-aware CV',
  evaluations: 25,
  groups: 69959,
  records: 70000,
  groupOverlap: 0,
  models: [
    { id: 'classical', name: 'Classical baseline', features: 8, accuracy: 0.7239, sensitivity: 0.6904, specificity: 0.7574, rocAuc: 0.7891, prAuc: 0.7738 },
    { id: 'dcqf', name: 'DCQF only', features: 24, accuracy: 0.7202, sensitivity: 0.6936, specificity: 0.7468, rocAuc: 0.7856, prAuc: 0.7679 },
    { id: 'hybrid_all', name: 'Hybrid — all DCQF', features: 32, accuracy: 0.7227, sensitivity: 0.6925, specificity: 0.7527, rocAuc: 0.7878, prAuc: 0.7710 },
    { id: 'hybrid_shap', name: 'Hybrid — SHAP-selected', features: 16, accuracy: 0.7221, sensitivity: 0.6929, specificity: 0.7513, rocAuc: 0.7876, prAuc: 0.7711 },
  ],
} as const

export const DCQF_ABLATION = {
  protocol: 'Stage B.3 — 5-fold × 5-repeat group-aware CV',
  models: [
    { id: 'classical', name: 'Classical', features: 8, accuracy: 0.7239, sensitivity: 0.6904, specificity: 0.7574, rocAuc: 0.7891, prAuc: 0.7738 },
    { id: 'quantum_singles', name: 'Quantum singles', features: 8, accuracy: 0.7189, sensitivity: 0.6928, specificity: 0.7450, rocAuc: 0.7852, prAuc: 0.7696 },
    { id: 'quantum_pairs', name: 'Quantum pairs', features: 8, accuracy: 0.7104, sensitivity: 0.6794, specificity: 0.7415, rocAuc: 0.7745, prAuc: 0.7592 },
    { id: 'quantum_triples', name: 'Quantum triples', features: 8, accuracy: 0.5942, sensitivity: 0.5145, specificity: 0.6737, rocAuc: 0.6322, prAuc: 0.6301 },
    { id: 'hybrid_singles', name: 'Hybrid + singles', features: 16, accuracy: 0.7230, sensitivity: 0.6950, specificity: 0.7510, rocAuc: 0.7882, prAuc: 0.7724 },
    { id: 'hybrid_pairs', name: 'Hybrid + pairs', features: 24, accuracy: 0.7230, sensitivity: 0.6938, specificity: 0.7521, rocAuc: 0.7880, prAuc: 0.7716 },
    { id: 'hybrid_all', name: 'Hybrid + all', features: 32, accuracy: 0.7227, sensitivity: 0.6925, specificity: 0.7527, rocAuc: 0.7878, prAuc: 0.7710 },
  ],
} as const

export const DCQF_SCREENING = {
  protocol: 'Stage B.4.1 — single group-aware 80/20 screening split; screening only',
  classical: { features: 8, accuracy: 0.7241, rocAuc: 0.7922, prAuc: 0.7748 },
  candidates: [
    { name: 'Top-8 quantum features', features: 16, accuracy: 0.7247, rocAuc: 0.7925, prAuc: 0.7736 },
    { name: 'Quantum singles', features: 16, accuracy: 0.7245, rocAuc: 0.7922, prAuc: 0.7739 },
    { name: 'Quantum pairs', features: 16, accuracy: 0.7255, rocAuc: 0.7924, prAuc: 0.7733 },
    { name: 'Singles + pairs', features: 24, accuracy: 0.7248, rocAuc: 0.7924, prAuc: 0.7736 },
    { name: 'All 24 DCQF', features: 32, accuracy: 0.7247, rocAuc: 0.7925, prAuc: 0.7731 },
  ],
} as const

export const DCQF_FEATURE_SCREENING = {
  protocol: 'Stage B.4.1 — training-only ranking on the 56,002-row training split',
  topFeatures: [
    { rank: 1, feature: 'Z0', trainAuc: 0.7336, strength: 0.2336 },
    { rank: 2, feature: 'Z0Z1', trainAuc: 0.3333, strength: 0.1667 },
    { rank: 3, feature: 'Z6', trainAuc: 0.6081, strength: 0.1081 },
    { rank: 4, feature: 'Z4', trainAuc: 0.5880, strength: 0.0880 },
    { rank: 5, feature: 'Z4Z5', trainAuc: 0.4190, strength: 0.0810 },
    { rank: 6, feature: 'Z4Z5Z6', trainAuc: 0.5514, strength: 0.0514 },
    { rank: 7, feature: 'Z1', trainAuc: 0.4507, strength: 0.0493 },
    { rank: 8, feature: 'Z5', trainAuc: 0.4525, strength: 0.0475 },
  ],
  note: 'The ranking is used only for candidate screening. It is not a replacement for the 5×5 group-aware benchmark.',
} as const

export const CLINICAL_REPRESENTATION = [
  ['Z0', 'ap_hi', 'Systolic blood pressure'],
  ['Z1', 'active', 'Physical activity'],
  ['Z2', 'smoke', 'Smoking'],
  ['Z3', 'gluc', 'Glucose'],
  ['Z4', 'age_years', 'Age'],
  ['Z5', 'bmi', 'BMI'],
  ['Z6', 'alco', 'Alcohol use'],
  ['Z7', 'gender', 'Gender'],
] as const

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
