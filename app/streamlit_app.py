"""Four screens. Resist building a fifth.

    streamlit run app/streamlit_app.py

The demo exists to make the benchmark legible in ninety seconds, not to be a product.
Every extra screen is time taken from the results table, and the results table is what
is being judged.

One rule enforced in code below: the app refuses to display a prediction without the
model's measured sensitivity and specificity next to it. A single-patient probability
shown alone, with no indication of how often the model is wrong, is the demo that gets
taken apart in questions -- and rightly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qheart import schema as S                                    # noqa: E402
from qheart.params import count_circuit_params, size_control_reducer  # noqa: E402

st.set_page_config(page_title="Hybrid QML - Heart Disease", layout="wide")
LEDGER = Path("results/runs/ledger.csv")

SCREENS = ["1 - Predict", "2 - Explain", "3 - Benchmark", "4 - Cost & Parameters"]
screen = st.sidebar.radio("Screen", SCREENS)
st.sidebar.caption("Four screens by design. A fifth costs results-table time.")


@st.cache_data
def load_ledger() -> pd.DataFrame | None:
    return pd.read_csv(LEDGER) if LEDGER.exists() else None


def model_reliability(led: pd.DataFrame | None, model: str) -> tuple[float, float] | None:
    if led is None or "model" not in led or model not in set(led["model"]):
        return None
    r = led[led["model"] == model]
    return float(r["sensitivity"].mean()), float(r["specificity"].mean())


led = load_ledger()

# ---------------------------------------------------------------- 1. Predict
if screen == SCREENS[0]:
    st.title("Predict")
    st.caption("Screening aid for demonstration. Not a diagnostic device, and not "
               "validated on any external cohort.")

    if led is None:
        st.warning("No results ledger yet. Run `make baselines` first -- this screen "
                   "deliberately will not show a prediction without measured "
                   "sensitivity and specificity to put beside it.")
        st.stop()

    models = sorted(set(led["model"]))
    model = st.selectbox("Model", models)
    rel = model_reliability(led, model)

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 20, 90, 54)
        max_hr = st.slider("Max heart rate achieved", 60, 210, 136)
        oldpeak = st.slider("ST depression (oldpeak, mm)", -2.0, 6.0, 0.9, 0.1)
    with c2:
        sex = st.selectbox("Sex", S.CATEGORIES["sex"], index=1)
        chest = st.selectbox("Chest pain type", S.CATEGORIES["chest_pain"], index=3)
        slope = st.selectbox("ST slope", S.CATEGORIES["st_slope"], index=1)
    with c3:
        angina = st.selectbox("Exercise-induced angina", S.CATEGORIES["exercise_angina"])
        ecg = st.selectbox("Resting ECG", S.CATEGORIES["resting_ecg"])
        chol = st.number_input("Cholesterol (mg/dl, 0 = not measured)", 0, 700, 240)

    if st.button("Predict", type="primary"):
        st.info("Wire this to a fitted model saved by `make baselines`. The stub is "
                "intentional: a demo that fabricates a probability is worse than one "
                "that says it is not connected yet.")
        if rel:
            se, sp = rel
            a, b = st.columns(2)
            a.metric("Model sensitivity", f"{se:.3f}",
                     help="Fraction of true cases detected, averaged over 25 folds")
            b.metric("Model specificity", f"{sp:.3f}",
                     help="Fraction of healthy patients correctly cleared")
            st.caption(f"At these rates, roughly {(1 - se) * 100:.0f} of every 100 "
                       f"patients with disease are missed. Show this number with every "
                       f"prediction.")
        if chol == 0:
            st.warning("Cholesterol 0 is a sentinel for 'not measured' in this dataset, "
                       "and which hospital a row came from is partly encoded by it. The "
                       "model may be reading provenance rather than physiology.")

# ---------------------------------------------------------------- 2. Explain
elif screen == SCREENS[1]:
    st.title("Explain")
    st.warning("**Two different explanations, never merged.** SHAP over a hybrid model "
               "explains the CLASSICAL HEAD -- its inputs are four expectation values, so "
               "it says nothing about the circuit. Parameter-shift saliency is the only "
               "figure here that explains the circuit itself.")
    a, b = st.columns(2)
    a.subheader("SHAP - classical path")
    a.caption("Scope: the head, or the full pipeline for purely classical models. "
              "Explains the MODEL, not the disease.")
    b.subheader("Parameter-shift saliency - quantum path")
    b.caption("Exact local derivative of the circuit output per input angle. A "
              "saturated response reads as zero importance, so read it as local.")
    st.info("Populate from `results/figures/` after `make compare`.")

# ---------------------------------------------------------------- 3. Benchmark
elif screen == SCREENS[2]:
    st.title("Benchmark")
    if led is None:
        st.warning("No ledger yet. `make baselines`.")
        st.stop()
    from qheart.report.tables import main_table

    tab = main_table(led)
    st.dataframe(tab.drop(columns=[c for c in tab.columns if c.endswith('__mean')]),
                 use_container_width=True)
    st.caption("mean +/- sd across 25 folds (5 repeats of stratified 5-fold). Folds share "
               "training data, so sd understates true uncertainty -- it is a spread, not a "
               "confidence interval.")
    if not any("control" in str(m) for m in set(led["model"])):
        st.error("No control_c row. Without the parameter-matched control the quantum "
                 "comparison is unfalsifiable -- the exact gap in both reference papers.")
    else:
        st.success("Parameter-matched control present. This is the comparison neither "
                   "reference paper runs.")

# ---------------------------------------------------------------- 4. Cost
else:
    st.title("Cost and parameters")
    c1, c2 = st.columns(2)
    with c1:
        nq = st.slider("Qubits", 2, 10, 4)
        depth = st.slider("Circuit depth", 1, 12, 6)
        cnt = count_circuit_params(nq, depth)
        st.metric("Trainable angles", cnt.trainable)
        st.metric("Model size at fp32", f"{cnt.bytes_fp32()} bytes")
        st.caption(f"{cnt.entanglers} entangling gates, which carry no trainable angle "
                   f"and are excluded. Reference paper P1 counts them, inflating 96 to 114.")
        st.caption(f"The VGG16 convolutional base is 14,714,688 parameters -- "
                   f"{14_714_688 // cnt.trainable:,}x larger. The full network is "
                   f"138,357,544 ({138_357_544 // cnt.trainable:,}x).")
    with c2:
        n = st.slider("Samples for the quantum kernel", 100, 3000, 918, 50)
        from qheart.quantum.budget import estimate_kernel_cost
        est = estimate_kernel_cost(n)
        st.metric("Circuit evaluations for the Gram matrix", f"{est.total_evals:,}")
        st.caption("O(N^2). This is why the dataset is 918 rows and not 20,000 -- a hard "
                   "constraint, not a shortcut.")
        st.code(size_control_reducer(cnt.trainable, 8, nq).summary(), language=None)
