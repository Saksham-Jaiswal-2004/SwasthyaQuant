from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.benchmark import router as benchmark_router
from app.api.routes.health import router as health_router
from app.api.routes.predict import router as predict_router
from app.services.model_service import model_service
from app.services.preprocessing_service import PreprocessingService

preprocessing_service = PreprocessingService()

app = FastAPI(
    title="QHeart Inference API",
    version="0.1.0",
    description=(
        "Production-style inference API for the hybrid qheart research pipeline. "
        "The backend reuses the repository's fitted research objects and does not "
        "train a replacement model on each request."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.preprocessing_service = preprocessing_service
app.state.model_service = model_service

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(benchmark_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "QHeart Hybrid Inference API",
        "status": "ready",
        "docs": "/docs",
    }
