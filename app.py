"""
COMP 3610 - Assignment 4
FastAPI application for serving NYC taxi tip predictions.

Model loading strategy:
  Load a local .pkl file (path from MODEL_PATH env var).

Run locally:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Docs:
  http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Configuration
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tip-api")

MODEL_PATH = os.getenv("MODEL_PATH", "rf_model.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "model_metrics.json")

FEATURE_NAMES = [
    "passenger_count", "log_trip_distance", "fare_amount",
    "pickup_hour", "pickup_day_of_week", "trip_duration_minutes",
    "trip_speed_mph", "is_weekend",
    "pickup_borough", "dropoff_borough",
]
MAX_BATCH_SIZE = 100

# Model container — singleton populated at startup, reused across requests
class ModelState:
    """Holds the loaded model and its metadata. Populated once on startup."""
    def __init__(self) -> None:
        self.model = None
        self.model_source: str = "not_loaded"
        self.model_name: str = "taxi-tip-regressor"
        self.model_version: str = "unknown"
        self.metrics: dict = {"mae": None, "rmse": None, "r2": None}
        self.loaded: bool = False


state = ModelState()


def _load_metrics() -> dict:
    """Load training metrics from a JSON file if present."""
    if Path(METRICS_PATH).exists():
        try:
            with open(METRICS_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read metrics file %s: %s", METRICS_PATH, e)
    return {"mae": None, "rmse": None, "r2": None}


def _load_from_disk() -> Optional[tuple]:
    """Attempt to load the model from a local .pkl file.
    Searches in order:
    1. Path specified in MODEL_PATH env var (absolute or relative to cwd)
    2. Current working directory
    3. Same directory as this script
    """
    # Try the configured path first
    path = Path(MODEL_PATH)
    if path.exists():
        logger.info("Loading model from disk: %s", path.resolve())
        model = joblib.load(path)
        mtime = int(path.stat().st_mtime)
        return model, f"local-{mtime}"
    
    # Try current working directory
    cwd_path = Path.cwd() / MODEL_PATH
    if cwd_path.exists():
        logger.info("Loading model from current directory: %s", cwd_path.resolve())
        model = joblib.load(cwd_path)
        mtime = int(cwd_path.stat().st_mtime)
        return model, f"local-{mtime}"
    
    # Try same directory as script
    script_dir = Path(__file__).parent / MODEL_PATH
    if script_dir.exists():
        logger.info("Loading model from script directory: %s", script_dir.resolve())
        model = joblib.load(script_dir)
        mtime = int(script_dir.stat().st_mtime)
        return model, f"local-{mtime}"
    
    logger.error("Model file not found at: %s, %s, or %s", path, cwd_path, script_dir)
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE at startup and keep it in the process memory."""
    logger.info("Starting application — loading model...")

    result = _load_from_disk()
    if result is None:
        logger.error(
            "No model could be loaded. /predict will return 503 until a model is available."
        )
    else:
        state.model, state.model_version = result
        state.model_source = "disk"
        state.loaded = True
        state.metrics = _load_metrics()
        logger.info(
            "Model loaded: name=%s version=%s source=%s",
            state.model_name, state.model_version, state.model_source,
        )

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="NYC Taxi Tip Prediction API",
    description="Serves tip_amount predictions from the registered regression model.",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic schemas — request validation and response shape
class TripFeatures(BaseModel):
    """Input features for a single trip. All fields required and constrained."""
    passenger_count: int = Field(..., ge=0, le=10, description="Number of passengers (0-10)")
    log_trip_distance: float = Field(..., ge=-5.0, le=10.0, description="log(trip_distance in miles)")
    fare_amount: float = Field(..., gt=0, le=500, description="Metered fare in USD (>0, <=500)")
    pickup_hour: int = Field(..., ge=0, le=23, description="Pickup hour 0-23")
    pickup_day_of_week: int = Field(..., ge=0, le=6, description="Monday=0 ... Sunday=6")
    trip_duration_minutes: float = Field(..., gt=0, le=1440, description="Minutes (>0, <=1440)")
    trip_speed_mph: float = Field(..., ge=0, le=200, description="Computed speed (0-200 mph)")
    is_weekend: bool = Field(..., description="True if Sat/Sun")
    pickup_borough: int = Field(..., ge=0, le=6, description="Label-encoded borough")
    dropoff_borough: int = Field(..., ge=0, le=6, description="Label-encoded borough")

    @field_validator("log_trip_distance")
    @classmethod
    def _check_log_distance_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("log_trip_distance must be a finite number")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "passenger_count": 1,
                "log_trip_distance": 0.95,
                "fare_amount": 12.5,
                "pickup_hour": 14,
                "pickup_day_of_week": 2,
                "trip_duration_minutes": 15.0,
                "trip_speed_mph": 10.3,
                "is_weekend": False,
                "pickup_borough": 3,
                "dropoff_borough": 3,
            }
        }
    }


class PredictionResponse(BaseModel):
    prediction_id: str = Field(..., description="UUID for traceability")
    tip_amount: float = Field(..., description="Predicted tip in USD, rounded to 2dp")
    model_version: str = Field(..., description="Version of the model that produced this prediction")


class BatchRequest(BaseModel):
    trips: list[TripFeatures] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    batch_id: str
    model_version: str
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    source: str
    feature_names: list[str]
    metrics: dict


class ErrorResponse(BaseModel):
    error: str
    detail: str
    prediction_id: Optional[str] = None



# Helpers
def _ensure_model_loaded() -> None:
    if not state.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs.",
        )


def _features_to_dataframe(features: list[TripFeatures]) -> pd.DataFrame:
    """Convert a list of validated TripFeatures into a DataFrame with the exact
    column order the pipeline expects."""
    rows = [f.model_dump() for f in features]
    df = pd.DataFrame(rows)[FEATURE_NAMES]
    # The training pipeline treats is_weekend as numeric (pandas bools behave as 0/1)
    df["is_weekend"] = df["is_weekend"].astype(int)
    return df


def _predict_many(features: list[TripFeatures]) -> np.ndarray:
    df = _features_to_dataframe(features)
    preds = state.model.predict(df)
    return np.asarray(preds).ravel()


# ──────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness + model-readiness probe."""
    return HealthResponse(
        status="ok" if state.loaded else "degraded",
        model_loaded=state.loaded,
        model_version=state.model_version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["meta"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    _ensure_model_loaded()
    return ModelInfoResponse(
        model_name=state.model_name,
        model_version=state.model_version,
        source=state.model_source,
        feature_names=FEATURE_NAMES,
        metrics=state.metrics,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict(trip: TripFeatures) -> PredictionResponse:
    """Predict tip_amount for a single trip."""
    _ensure_model_loaded()
    pred = float(_predict_many([trip])[0])
    return PredictionResponse(
        prediction_id=str(uuid.uuid4()),
        tip_amount=round(pred, 2),
        model_version=state.model_version,
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
def predict_batch(payload: BatchRequest) -> BatchResponse:
    """Predict tip_amount for up to 100 trips in a single request."""
    _ensure_model_loaded()
    preds = _predict_many(payload.trips)
    predictions = [
        PredictionResponse(
            prediction_id=str(uuid.uuid4()),
            tip_amount=round(float(p), 2),
            model_version=state.model_version,
        )
        for p in preds
    ]
    return BatchResponse(
        batch_id=str(uuid.uuid4()),
        model_version=state.model_version,
        count=len(predictions),
        predictions=predictions,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    incident_id = str(uuid.uuid4())
    # Full traceback goes to logs only, not to the client.
    logger.error(
        "Unhandled error on %s %s (incident=%s):\n%s",
        request.method, request.url.path, incident_id, traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred. Contact support with the incident_id.",
            "incident_id": incident_id,
        },
    )