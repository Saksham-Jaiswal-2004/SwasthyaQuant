from __future__ import annotations

from pathlib import Path

from app.config import settings


class BenchmarkService:
    def get_benchmark(self) -> dict[str, object]:
        ledger_path = settings.project_root / "results" / "runs" / "ledger.csv"
        if not ledger_path.exists():
            return {"status": "unavailable", "message": "No benchmark ledger was found."}

        rows = []
        with ledger_path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                values = [v.strip() for v in line.split(",")]
                if len(values) != len(header):
                    continue
                rows.append(dict(zip(header, values)))

        models: dict[str, object] = {}
        for row in rows:
            key = row.get("model")
            if key is None:
                continue
            models.setdefault(key, []).append(row)

        return {
            "status": "available",
            "models": {name: payload[-1] for name, payload in models.items()},
        }


benchmark_service = BenchmarkService()
