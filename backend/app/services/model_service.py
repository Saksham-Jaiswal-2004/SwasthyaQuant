from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings


class ModelArtifactsUnavailableError(RuntimeError):
    """Raised when the repository does not include a frozen inference artifact."""


class ModelService:
    """Loads a single frozen inference bundle once and reuses it in memory."""

    ModelArtifactsUnavailableError = ModelArtifactsUnavailableError

    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None
        self._load_attempted = False
        self.ModelArtifactsUnavailableError = ModelArtifactsUnavailableError

    def _candidate_paths(self) -> list[Path]:
        candidates: list[Path] = []
        for name in settings.model_artifact_names:
            candidates.append(settings.artifact_dir / name)
            candidates.append(settings.project_root / "backend" / "artifacts" / name)
            candidates.append(settings.project_root / "artifacts" / name)
            candidates.append(settings.project_root / name)
        return list(dict.fromkeys(candidates))

    def _build_fallback_bundle(self) -> dict[str, Any]:
        """Generate a working bundle from the repo’s own dataset and preprocessing pipeline."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler

            from qheart import schema as S
            from qheart.data import load
            from qheart.preprocess.pipeline import ClinicalRepresentation
            from qheart.quantum.dcqf import DCQFExtractor
        except Exception as exc:  # pragma: no cover - defensive guard for missing runtime deps
            raise ModelArtifactsUnavailableError(
                "The backend cannot build a fallback model because the training dependencies are unavailable."
            ) from exc

        data_path = settings.project_root / "data" / "raw" / "cardio_train.csv"
        if not data_path.exists():
            raise ModelArtifactsUnavailableError(
                "No training dataset was found at data/raw/cardio_train.csv."
            )

        df = load("cardio_70000", data_path)
        X = df[S.FEATURES]
        y = df[S.TARGET].to_numpy(dtype=int)

        # Keep startup fast but still trained on the repository’s real feature pipeline.
        sample_size = min(len(df), 10000)
        X = X.iloc[:sample_size].copy()
        y = y[:sample_size]

        clinical_representation = ClinicalRepresentation(
            k=8,
            redundancy_penalty=0.5,
            output="angles",
            random_state=20260830,
        )
        clinical_representation.fit(X, y)
        X8 = clinical_representation.transform(X)

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
        dcqf.fit(X8, y)
        Q = dcqf.transform(X8)
        hybrid = np.concatenate([X8, Q], axis=1)

        scaler = StandardScaler()
        hybrid_scaled = scaler.fit_transform(hybrid)

        model = GradientBoostingClassifier(
            random_state=20260830,
            n_estimators=80,
            learning_rate=0.05,
            max_depth=3,
        )
        model.fit(hybrid_scaled, y)

        bundle = {
            "clinical_representation": clinical_representation,
            "selector": clinical_representation.selector_,
            "dcqf": dcqf,
            "scaler": scaler,
            "model": model,
        }

        target_dir = settings.artifact_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = target_dir / "inference_bundle.joblib"
        with bundle_path.open("wb") as handle:
            pickle.dump(bundle, handle)

        return bundle

    def _load_bundle(self) -> dict[str, Any]:
        for path in self._candidate_paths():
            if not path.exists():
                continue
            try:
                with path.open("rb") as handle:
                    bundle = pickle.load(handle)
                if isinstance(bundle, dict):
                    return bundle
            except Exception:
                continue

        try:
            return self._build_fallback_bundle()
        except ModelArtifactsUnavailableError:
            raise ModelArtifactsUnavailableError(
                "No frozen inference artifact was found. The repository contains research CV logs but no trained pipeline bundle. "
                "Export the trained ClinicalRepresentation, feature selector, DCQF state, StandardScaler, and GradientBoostingClassifier "
                "into a serialized artifact bundle before starting the backend."
            )

    def load(self) -> dict[str, Any]:
        if self._bundle is not None:
            return self._bundle
        self._bundle = self._load_bundle()
        self._load_attempted = True
        return self._bundle

    def get_model(self) -> Any:
        bundle = self.load()
        if "model" not in bundle:
            raise ModelArtifactsUnavailableError(
                "Model artifact bundle is missing the fitted classifier under the 'model' key."
            )
        return bundle["model"]

    def get_scaler(self) -> Any:
        bundle = self.load()
        if "scaler" not in bundle:
            raise ModelArtifactsUnavailableError(
                "Artifact bundle is missing the fitted StandardScaler under the 'scaler' key."
            )
        return bundle["scaler"]

    def get_selector(self) -> Any:
        bundle = self.load()
        if "selector" not in bundle:
            raise ModelArtifactsUnavailableError(
                "Artifact bundle is missing the fitted feature selector under the 'selector' key."
            )
        return bundle["selector"]

    def get_clinical_representation(self) -> Any:
        bundle = self.load()
        if "clinical_representation" not in bundle:
            raise ModelArtifactsUnavailableError(
                "Artifact bundle is missing the fitted ClinicalRepresentation under the 'clinical_representation' key."
            )
        return bundle["clinical_representation"]

    def get_dcqf(self) -> Any:
        bundle = self.load()
        if "dcqf" not in bundle:
            raise ModelArtifactsUnavailableError(
                "Artifact bundle is missing the fitted DCQF extractor under the 'dcqf' key."
            )
        return bundle["dcqf"]

    def is_loaded(self) -> bool:
        try:
            self.load()
            return True
        except ModelArtifactsUnavailableError:
            return False


model_service = ModelService()
