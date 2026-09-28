import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
SRC = ROOT / "src"

for candidate in (str(ROOT), str(BACKEND), str(SRC)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)
