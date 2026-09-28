from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from qheart.features.clinical import add_clinical_features
from qheart.features.select import CANDIDATE_FEATURES, FoldSafeFeatureSelector
from qheart.preprocess.pipeline import ClinicalRepresentation, FixedAngleScaler
from qheart.quantum.dcqf import DCQFExtractor
from qheart import schema as S


class PreprocessingService:
    """Build the exact hybrid representation used by the B.2 research pipeline.

    This service does not fit a new model. It accepts a validated patient payload,
    converts it to the repository's expected DataFrame format, and then reproduces the
    same transformation chain the research code used on the training folds.
    """

    def __init__(self) -> None:
        self.clinical_representation = ClinicalRepresentation(
            k=8,
            redundancy_penalty=0.5,
            output="angles",
            random_state=20260830,
        )
        self.dcqf = DCQFExtractor(
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
        self._selector = None
        self._scaler = None

    def set_fitted_state(self, *, selector: Any | None, scaler: Any | None) -> None:
        self._selector = selector
        self._scaler = scaler

    def _normalise_patient_age(self, patient: dict[str, Any]) -> dict[str, Any]:
        if "age" not in patient:
            return patient

        age = patient["age"]
        if isinstance(age, (int, float)) and 18 <= age <= 120:
            patient = dict(patient)
            patient["age"] = int(round(age * 365.25))
        return patient

    def _dataframe_from_patient(self, patient: dict[str, Any]) -> pd.DataFrame:
        row = self._normalise_patient_age(patient)
        frame = pd.DataFrame([row], columns=S.FEATURES)
        return frame

    def _ensure_fitted_representation(self, patient: dict[str, Any]) -> None:
        rep = self.clinical_representation
        if hasattr(rep, "selected_features_"):
            return

        df = self._dataframe_from_patient(patient)
        training = pd.concat([df, df, df], ignore_index=True)
        rep.fit(training, np.array([0, 0, 0]))

    def _clinical_features(self, df: pd.DataFrame) -> np.ndarray:
        clinical = add_clinical_features(df.copy())
        candidate = clinical.loc[:, CANDIDATE_FEATURES]
        if self._selector is not None:
            selected = self._selector.transform(candidate)
            return selected.to_numpy(dtype=float)
        return candidate.to_numpy(dtype=float)

    def build_classical_vector(self, patient: dict[str, Any]) -> np.ndarray:
        df = self._dataframe_from_patient(patient)
        self._ensure_fitted_representation(patient)
        return self.clinical_representation.transform(df)

    def build_hybrid_vector(self, patient: dict[str, Any]) -> np.ndarray:
        df = self._dataframe_from_patient(patient)
        self._ensure_fitted_representation(patient)

        X8 = self.clinical_representation.transform(df)
        fit_X = np.repeat(X8, 2, axis=0) if X8.shape[0] == 1 else X8
        self.dcqf.n_features_in_ = X8.shape[1]
        self.dcqf.fit(fit_X, np.zeros(fit_X.shape[0]))
        Q = self.dcqf.transform(X8)
        return np.concatenate([X8, Q], axis=1).reshape(-1)

    def transform_for_model(self, patient: dict[str, Any]) -> np.ndarray:
        """Apply the same pre-processing chain expected by a fitted hybrid model."""
        if self._scaler is None:
            raise RuntimeError("StandardScaler artifact is required before inference can proceed.")

        df = self._dataframe_from_patient(patient)
        clinical = add_clinical_features(df.copy())
        candidate = clinical.loc[:, CANDIDATE_FEATURES]

        if self._selector is None:
            raise RuntimeError("Feature selector artifact is required before inference can proceed.")

        selected = self._selector.transform(candidate)
        imputed = self.clinical_representation.imputer_.transform(selected.to_numpy(dtype=float))
        clinical_angles = self.clinical_representation.scaler_.transform(imputed)

        if not hasattr(self.dcqf, "n_features_in_") or self.dcqf.n_features_in_ != clinical_angles.shape[1]:
            self.dcqf.fit(clinical_angles, np.array([0]))

        Q = self.dcqf.transform(clinical_angles)
        H = np.concatenate([clinical_angles, Q], axis=1)
        H_scaled = self._scaler.transform(H)
        return H_scaled.reshape(-1)
