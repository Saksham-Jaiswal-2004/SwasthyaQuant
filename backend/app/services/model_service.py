from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

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
