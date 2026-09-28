import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qheart.data.loaders import load
from qheart.models.dcqf_control import ChainProductControl, ScrambledDCQF
from qheart.preprocess.pipeline import ClinicalRepresentation
from qheart.quantum.dcqf import DCQFExtractor


SEED = 20260830
N_SAMPLES = 1000


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def evaluate(name, X_train, X_test, y_train, y_test):
    """
    Fit the same classical head for every representation.

    Scaling is fitted on the training split only.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )

    model.fit(X_train_s, y_train)

    proba = model.predict_proba(X_test_s)[:, 1]
    pred = (proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "model": name,
        "n_features": X_train.shape[1],
        "accuracy": accuracy_score(y_test, pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }


# ------------------------------------------------------------
# Load real 70k cardiovascular dataset
# ------------------------------------------------------------
df = load(
    "cardio_70000",
    "data/raw/cardio_train.csv",
)

y_all = df["cardio"].to_numpy(dtype=int)
X_raw_all = df.drop(columns=["cardio"])


# ------------------------------------------------------------
# Small controlled sample
# ------------------------------------------------------------
rng = np.random.default_rng(SEED)

indices = rng.choice(
    len(df),
    size=N_SAMPLES,
    replace=False,
)

X_raw = X_raw_all.iloc[indices].reset_index(drop=True)
y = y_all[indices]


# ------------------------------------------------------------
# One fixed stratified train/test split
# ------------------------------------------------------------
train_idx, test_idx = train_test_split(
    np.arange(N_SAMPLES),
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

X_train_raw = X_raw.iloc[train_idx].copy()
X_test_raw = X_raw.iloc[test_idx].copy()

y_train = y[train_idx]
y_test = y[test_idx]


print("=" * 72)
print("STAGE B — DCQF VS MATCHED CONTROLS")
print("=" * 72)

print(f"Total samples: {N_SAMPLES}")
print(f"Training samples: {len(train_idx)}")
print(f"Test samples: {len(test_idx)}")
print(f"Training positives: {int(y_train.sum())}")
print(f"Test positives: {int(y_test.sum())}")


# ------------------------------------------------------------
# Clinical representation
#
# IMPORTANT:
# Fit feature selection on the training split only.
# ------------------------------------------------------------
rep = ClinicalRepresentation(
    k=8,
    redundancy_penalty=0.5,
    output="angles",
    random_state=SEED,
)

X8_train = rep.fit_transform(
    X_train_raw,
    y_train,
)

X8_test = rep.transform(
    X_test_raw,
)

print("\nClinical representation:")
print("  train:", X8_train.shape)
print("  test :", X8_test.shape)


# ------------------------------------------------------------
# 1. Classical 8-feature baseline
# ------------------------------------------------------------
results = []

results.append(
    evaluate(
        "classical_8",
        X8_train,
        X8_test,
        y_train,
        y_test,
    )
)


# ------------------------------------------------------------
# 2. DCQF
#
# Couplings are fitted on TRAINING DATA ONLY.
# ------------------------------------------------------------
print("\nFitting DCQF...")

dcqf = DCQFExtractor(
    orders_encoded=(2,),
    orders_read=(1, 2, 3),
    n_bins=4,
    shots=None,
    trotter_steps=1,
    dt=1.0,
    agp_scale=1.0,
    include_input=False,
    seed=SEED,
)

Zq_train = dcqf.fit_transform(
    X8_train,
    y_train,
)

Zq_test = dcqf.transform(
    X8_test,
)

print("  train:", Zq_train.shape)
print("  test :", Zq_test.shape)
print("  parameters:", dcqf.n_trainable_params())

results.append(
    evaluate(
        "dcqf_24",
        Zq_train,
        Zq_test,
        y_train,
        y_test,
    )
)


# ------------------------------------------------------------
# 3. ChainProduct dimension-matched classical control
# ------------------------------------------------------------
print("\nFitting ChainProduct control...")

product = ChainProductControl(
    orders=(1, 2, 3),
    squash=True,
)

Zp_train = product.fit_transform(
    X8_train,
    y_train,
)

Zp_test = product.transform(
    X8_test,
)

print("  train:", Zp_train.shape)
print("  test :", Zp_test.shape)
print("  parameters:", product.n_trainable_params())

results.append(
    evaluate(
        "product_control_24",
        Zp_train,
        Zp_test,
        y_train,
        y_test,
    )
)


# ------------------------------------------------------------
# 4. Scrambled DCQF null model
# ------------------------------------------------------------
print("\nFitting Scrambled DCQF...")

scrambled = ScrambledDCQF(
    orders_encoded=(2,),
    orders_read=(1, 2, 3),
    n_bins=4,
    shots=None,
    trotter_steps=1,
    dt=1.0,
    agp_scale=1.0,
    include_input=False,
    seed=SEED,
)

Zs_train = scrambled.fit_transform(
    X8_train,
    y_train,
)

Zs_test = scrambled.transform(
    X8_test,
)

print("  train:", Zs_train.shape)
print("  test :", Zs_test.shape)
print("  parameters:", scrambled.n_trainable_params())

results.append(
    evaluate(
        "scrambled_dcqf_24",
        Zs_train,
        Zs_test,
        y_train,
        y_test,
    )
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
print("\n" + "=" * 72)
print("RESULTS")
print("=" * 72)

header = (
    f"{'Model':<24}"
    f"{'Feat':>6}"
    f"{'Acc':>9}"
    f"{'Sens':>9}"
    f"{'Spec':>9}"
    f"{'ROC-AUC':>10}"
    f"{'PR-AUC':>10}"
)

print(header)
print("-" * len(header))

for r in results:
    print(
        f"{r['model']:<24}"
        f"{r['n_features']:>6}"
        f"{r['accuracy']:>9.4f}"
        f"{r['sensitivity']:>9.4f}"
        f"{r['specificity']:>9.4f}"
        f"{r['roc_auc']:>10.4f}"
        f"{r['pr_auc']:>10.4f}"
    )


# ------------------------------------------------------------
# Simple diagnostic comparisons
# ------------------------------------------------------------
print("\n" + "=" * 72)
print("DIAGNOSTIC COMPARISONS")
print("=" * 72)

by_name = {r["model"]: r for r in results}

base = by_name["classical_8"]
q = by_name["dcqf_24"]
prod = by_name["product_control_24"]
scr = by_name["scrambled_dcqf_24"]

print(
    f"DCQF PR-AUC - classical: "
    f"{q['pr_auc'] - base['pr_auc']:+.4f}"
)

print(
    f"DCQF PR-AUC - product control: "
    f"{q['pr_auc'] - prod['pr_auc']:+.4f}"
)

print(
    f"DCQF PR-AUC - scrambled DCQF: "
    f"{q['pr_auc'] - scr['pr_auc']:+.4f}"
)

print("\nInterpretation:")
print(
    "  DCQF > product control:"
    " evidence that the quantum map adds something beyond"
    " the matched classical feature expansion."
)
print(
    "  DCQF > scrambled DCQF:"
    " evidence that the data-derived MI coupling structure"
    " contributes to performance."
)
print(
    "  DCQF > both:"
    " strongest result for the DCQF feature-map hypothesis."
)
print(
    "  No improvement:"
    " valid negative result; do not claim quantum advantage."
)