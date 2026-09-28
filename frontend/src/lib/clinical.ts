import type { PatientInput } from '../types/api'

// Display-only mirror of the deterministic formulas in src/qheart/features/clinical.py.
// The backend recomputes these itself; nothing here feeds the model.

export interface DerivedFeatures {
  bmi: number | null
  bmiClass: 'Underweight' | 'Normal' | 'Overweight' | 'Obese' | null
  pulsePressure: number | null
  map: number | null
  bpGroup: 'Normal' | 'Elevated' | 'Stage 1' | 'Stage 2' | 'Crisis' | null
  ageGroup: 'Young' | 'Middle' | 'Older' | 'Senior' | 'Elderly' | null
}

const between = (v: number | undefined, lo: number, hi: number) =>
  typeof v === 'number' && Number.isFinite(v) && v >= lo && v <= hi

export function deriveFeatures(p: Partial<PatientInput>): DerivedFeatures {
  const hOk = between(p.height, 100, 250)
  const wOk = between(p.weight, 20, 300)
  let hiOk = between(p.ap_hi, 60, 250)
  let loOk = between(p.ap_lo, 30, 150)
  if (hiOk && loOk && p.ap_hi! < p.ap_lo!) hiOk = loOk = false

  let bmi: number | null = hOk && wOk ? p.weight! / (p.height! / 100) ** 2 : null
  if (bmi !== null && !(bmi >= 10 && bmi <= 80)) bmi = null

  const pulsePressure = hiOk && loOk ? p.ap_hi! - p.ap_lo! : null
  const map = hiOk && loOk ? p.ap_lo! + (p.ap_hi! - p.ap_lo!) / 3 : null

  const bmiClass = bmi === null ? null
    : bmi < 18.5 ? 'Underweight' : bmi < 25 ? 'Normal' : bmi < 30 ? 'Overweight' : 'Obese'
  const bpGroup = !hiOk ? null
    : p.ap_hi! < 120 ? 'Normal' : p.ap_hi! < 130 ? 'Elevated' : p.ap_hi! < 140 ? 'Stage 1' : p.ap_hi! < 180 ? 'Stage 2' : 'Crisis'
  const a = p.age
  const ageGroup = typeof a !== 'number' || !Number.isFinite(a) ? null
    : a < 40 ? 'Young' : a < 50 ? 'Middle' : a < 60 ? 'Older' : a < 70 ? 'Senior' : 'Elderly'

  return { bmi, bmiClass, pulsePressure, map, bpGroup, ageGroup }
}

export const LEVEL_LABELS: Record<1 | 2 | 3, string> = { 1: 'Normal', 2: 'Above normal', 3: 'Well above normal' }
export const GENDER_LABELS: Record<1 | 2, string> = { 1: 'Female', 2: 'Male' }
