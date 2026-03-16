"""FastAPI model serving endpoint."""
from __future__ import annotations

import io
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.deployment.inference import InferencePipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="MLOps-Platform Model Serving",
    description="Image classification inference API",
    version="1.2.0",
)

# Global inference pipeline (initialized on startup)
pipeline: Optional[InferencePipeline] = None


class PredictionResponse(BaseModel):
    class_name: str
    class_index: int
    confidence: float
    top_k: list[dict]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None


@app.on_event("startup")
async def startup():
    global pipeline
    import os
    model_path = os.environ.get("MODEL_PATH", "models/saved/best.pt")
    try:
        pipeline = InferencePipeline(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if pipeline is not None else "degraded",
        model_loaded=pipeline is not None,
        model_name=pipeline.model_name if pipeline else None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), top_k: int = 5):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        start = time.perf_counter()
        result = pipeline.predict(io.BytesIO(contents), top_k=top_k)
        latency = (time.perf_counter() - start) * 1000

        return PredictionResponse(
            class_name=result["class_name"],
            class_index=result["class_index"],
            confidence=result["confidence"],
            top_k=result["top_k"],
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
