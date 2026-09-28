from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from app.config import settings

router = APIRouter(prefix="/api", tags=["benchmark"])


def _read_ledger() -> dict[str, object]:
    ledger_path = settings.project_root / "results" / "runs" / "ledger.csv"
    if not ledger_path.exists():
        return {
            "status": "unavailable",
            "message": "No research benchmark ledger was found under results/runs/ledger.csv.",
            "models": {},
        }

    rows = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(header):
                continue
            record = dict(zip(header, parts))
            rows.append(record)

    models: dict[str, dict[str, object]] = {}
    for row in rows:
        name = row.get("model")
        if not name:
            continue
        if name not in models:
            models[name] = {"count": 0, "rows": []}
        models[name]["count"] = int(models[name]["count"]) + 1
        models[name]["rows"].append(row)

    for name, payload in list(models.items()):
        payload["latest"] = payload["rows"][-1]
        payload.pop("rows")

    return {
        "status": "available",
        "models": models,
        "source": str(ledger_path.relative_to(settings.project_root)),
    }


@router.get("/benchmark")
async def benchmark() -> dict[str, object]:
    return _read_ledger()
