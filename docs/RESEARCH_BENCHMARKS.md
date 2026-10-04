# Swasthya Quant — Research Benchmark Snapshot

This file is the source-of-truth summary used by the prototype UI. These numbers are research benchmarks, not clinical validation. The backend inference artifact and these cross-validation summaries are deliberately kept separate: the UI must not imply that a request-time prediction re-runs the 25 CV folds.

## Evidence hierarchy

1. **Primary result:** Stage B.2, 5-fold × 5-repeat group-aware CV.
2. **Ablation:** Stage B.3, same group-aware evaluation used to test DCQF observable order.
3. **Screening:** Stage B.4/B.4.1, one group-aware split used only to narrow candidates; it is not a final benchmark.
4. **Controlled research test:** synthetic DCQF null experiment; it is not cardiovascular evidence.

## Stage B.2 — Hybrid DCQF

- Dataset: 70,000 records; 34,979 positive / 35,021 negative.
- Unique feature groups: 69,959.
- Validation: 5 folds × 5 repeats, group-aware; 25 evaluations.
- Group overlap: 0 between train and test in each split.
- Clinical representation: 8 selected features.
- DCQF representation: 24 features = 8 singles + 8 pairs + 8 triples.
- Hybrid vector: 32 features.
- SHAP selection: top-K = 16, selected inside each training fold.

| Model | Features | Accuracy | Sensitivity | Specificity | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Classical | 8 | 0.7239 | 0.6904 | 0.7574 | 0.7891 | 0.7738 |
| DCQF only | 24 | 0.7202 | 0.6936 | 0.7468 | 0.7856 | 0.7679 |
| Hybrid all | 32 | 0.7227 | 0.6925 | 0.7527 | 0.7878 | 0.7710 |
| Hybrid SHAP | 16 | 0.7221 | 0.6929 | 0.7513 | 0.7876 | 0.7711 |

Interpretation: the classical baseline is slightly ahead overall. The hybrid representation is competitive but does not establish quantum advantage.

## Stage B.3 — DCQF feature-order ablation

| Configuration | Features | Accuracy | Sensitivity | Specificity | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Classical | 8 | 0.7239 | 0.6904 | 0.7574 | 0.7891 | 0.7738 |
| Quantum singles | 8 | 0.7189 | 0.6928 | 0.7450 | 0.7852 | 0.7696 |
| Quantum pairs | 8 | 0.7104 | 0.6794 | 0.7415 | 0.7745 | 0.7592 |
| Quantum triples | 8 | 0.5942 | 0.5145 | 0.6737 | 0.6322 | 0.6301 |
| Hybrid + singles | 16 | 0.7230 | 0.6950 | 0.7510 | 0.7882 | 0.7724 |
| Hybrid + pairs | 24 | 0.7230 | 0.6938 | 0.7521 | 0.7880 | 0.7716 |
| Hybrid + all | 32 | 0.7227 | 0.6925 | 0.7527 | 0.7878 | 0.7710 |

Key finding: triples are substantially weaker as a standalone feature family. Adding singles, pairs, or triples to the classical representation did not improve ROC-AUC over the classical baseline.

## Stage B.4 — Fast screening

Single group-aware 80/20 split. This stage is for candidate screening only and must not be presented as the final benchmark.

| Configuration | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Classical | 0.7241 | 0.7922 | 0.7748 |
| Top-8 quantum + classical | 0.7247 | 0.7925 | 0.7736 |
| Quantum singles + classical | 0.7245 | 0.7922 | 0.7739 |
| Quantum pairs + classical | 0.7255 | 0.7924 | 0.7733 |
| Singles + pairs + classical | 0.7248 | 0.7924 | 0.7736 |
| All 24 DCQF + classical | 0.7247 | 0.7925 | 0.7731 |

### Stage B.4.1 — training-only feature ranking

The training split ranked the following eight DCQF observables highest by absolute training AUC distance from 0.5: Z0 (0.7336), Z0Z1 (0.3333), Z6 (0.6081), Z4 (0.5880), Z4Z5 (0.4190), Z4Z5Z6 (0.5514), Z1 (0.4507), and Z5 (0.4525). This ranking was used for screening only and must not be interpreted as held-out evidence of generalization.

## Quantum-specific null test

Synthetic controlled experiment: n=400, 4 seeds × 5 folds, 35% label noise.

DCQF vs scrambled null AUC difference:
- Linear: -0.0139, not significant.
- Pairwise: +0.0135, not significant.
- Mixed: -0.0080, not significant.

DCQF vs chain-product control:
- Linear: -0.1373, significant against DCQF.
- Pairwise: +0.0058, not significant.
- Mixed: -0.1114, significant against DCQF.

This experiment does not prove DCQF is ineffective; it shows no detectable quantum-specific contribution at this scale.

## Deployment wording

The prototype backend uses the architecture:

Raw clinical input → fold-safe clinical representation → 8 angles → 8-qubit DCQF → 24 quantum observables → 32-feature hybrid vector → StandardScaler → GradientBoostingClassifier → risk probability.

The application is a research/decision-support prototype and is not a medical diagnostic device. The current evidence does not demonstrate quantum advantage; the scientific contribution is the controlled, leakage-safe evaluation of whether DCQF-derived representations add value to the classical baseline.
