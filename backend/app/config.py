from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
BACKEND_ROOT = Path(__file__).resolve().parents[1]

for candidate in (str(SRC_ROOT), str(PROJECT_ROOT), str(BACKEND_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

ARTIFACT_DIR = Path(
    os.getenv("ARTIFACT_DIR", str(BACKEND_ROOT / "artifacts"))
).expanduser()


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    source_root: Path = SRC_ROOT
    backend_root: Path = BACKEND_ROOT
    artifact_dir: Path = ARTIFACT_DIR
    model_artifact_names: tuple[str, ...] = (
        "inference_bundle.joblib",
        "hybrid_inference_bundle.joblib",
        "inference_bundle.pkl",
        "hybrid_inference_bundle.pkl",
    )


settings = Settings()
