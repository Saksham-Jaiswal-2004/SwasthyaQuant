from __future__ import annotations

from fastapi import APIRouter

from app.services.model_service import model_service

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health() -> dict[str, object]:
    is_loaded = model_service.is_loaded()
    return {
        "status": "ok" if is_loaded else "degraded",
        "model_loaded": is_loaded,
    }
