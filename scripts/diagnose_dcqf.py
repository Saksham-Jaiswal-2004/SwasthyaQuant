import numpy as np

from qheart.data.loaders import load
from qheart.preprocess.pipeline import ClinicalRepresentation
from qheart.quantum.dcqf import DCQFExtractor


# ------------------------------------------------------------
# Load the real 70k cardiovascular dataset
# ------------------------------------------------------------
df = load(
    "cardio_70000",
    "data/raw/cardio_train.csv",
)

y = df["cardio"].to_numpy(dtype=int)
X_raw = df.drop(columns=["cardio"])


# ------------------------------------------------------------
# Build the same clinical representation used by the pipeline
# ------------------------------------------------------------
rep = ClinicalRepresentation(
    k=8,
    redundancy_penalty=0.5,
    output="angles",
    random_state=20260830,
)


# Small diagnostic subset only.
# This is NOT a predictive experiment.
X_small = X_raw.iloc[:500].copy()
y_small = y[:500]

X8 = rep.fit_transform(X_small, y_small)

print("Input shape:", X8.shape)
print(
    "Input range:",
    float(np.min(X8)),
    "to",
    float(np.max(X8)),
)


# ------------------------------------------------------------
# DCQF
# ------------------------------------------------------------
dcqf = DCQFExtractor(
    orders_encoded=(2,),
    orders_read=(1, 2, 3),
    n_bins=4,
    shots=None,
    trotter_steps=1,
    dt=1.0,
    agp_scale=1.0,
    include_input=False,
    seed=20260830,
)

dcqf.fit(X8, y_small)

print("\nSelected feature count:", X8.shape[1])
print("DCQF output features:", dcqf.n_output_features())
print("Trainable quantum parameters:", dcqf.n_trainable_params())


print("\nCircuit cost:")
print(dcqf.circuit_cost())


print("\nDiagnostics:")
print(dcqf.diagnostics(X8[:100]))


# ------------------------------------------------------------
# Transform diagnostic samples
# ------------------------------------------------------------
Z = dcqf.transform(X8[:100])

print("\nQuantum feature matrix:", Z.shape)
print("Finite:", bool(np.all(np.isfinite(Z))))
print("Mean:", float(np.mean(Z)))

# Global standard deviation across all samples and features.
print("Global std:", float(np.std(Z)))

# Mean of the standard deviation of each individual feature.
print(
    "Mean per-feature std:",
    float(np.mean(np.std(Z, axis=0))),
)

print("Abs max:", float(np.max(np.abs(Z))))


print("\nFirst sample:")
print(Z[0])


print("\nFeature names:")
print(dcqf.feature_names())


print("\nProvenance:")
print(dcqf.provenance())