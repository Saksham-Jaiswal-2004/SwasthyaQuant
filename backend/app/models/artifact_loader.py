from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from app.config import settings


class ArtifactLoader:
    """Discover and deserialize the frozen inference bundle.

    The repository does not currently include such a bundle. This loader exists to
    enforce the export contract and fail loudly when the required trained objects are
    absent.
    """

    def __init__(self, artifact_dir: str | Path | None = None):
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else settings.artifact_dir

    def candidate_paths(self) -> list[Path]:
        base_candidates = [
            self.artifact_dir,
            settings.project_root / "backend" / "artifacts",
            settings.project_root / "artifacts",
            settings.project_root,
        ]
        names = settings.model_artifact_names
        paths: list[Path] = []
        for base in base_candidates:
            for name in names:
                paths.append(Path(base) / name)
        return list(dict.fromkeys(paths))

    def load_bundle(self) -> dict[str, Any]:
        for path in self.candidate_paths():
            if not path.exists():
                continue
            try:
                with path.open("rb") as handle:
                    bundle = pickle.load(handle)
                if isinstance(bundle, dict):
                    return bundle
            except Exception:
                continue
        raise FileNotFoundError(
            "No frozen inference bundle found. Export the trained ClinicalRepresentation, "
            "feature selector, DCQF state, StandardScaler, and GradientBoostingClassifier "
            "into one pickle/joblib bundle under backend/artifacts or a configured artifact dir."
        )
