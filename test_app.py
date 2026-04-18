"""
COMP 3610 - Assignment 4
Test suite for the FastAPI tip prediction service.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

VALID_TRIP = {
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

FEATURE_NAMES = [
    "passenger_count", "log_trip_distance", "fare_amount",
    "pickup_hour", "pickup_day_of_week", "trip_duration_minutes",
    "trip_speed_mph", "is_weekend",
    "pickup_borough", "dropoff_borough",
]


# ──────────────────────────────────────────────────────────────────────────
# Stub model — ensures tests run even without a trained .pkl on disk
# ──────────────────────────────────────────────────────────────────────────
class _StubModel:
    """A tiny replacement for the real pipeline. Returns fare_amount * 0.2 as
    the 'predicted tip'. Implements only the .predict interface the app uses."""
    feature_names_in_ = FEATURE_NAMES

    def predict(self, df):
        fare = df["fare_amount"].to_numpy(dtype=float)
        return fare * 0.2


@pytest.fixture(scope="session", autouse=True)
def ensure_model_available(tmp_path_factory):
    """Write a stub model + metrics file before the app starts up, so that the
    lifespan handler has something to load. If the real artifacts are already
    present in the working directory, we leave them alone."""
    stub_dir = tmp_path_factory.mktemp("model_artifacts")
    stub_model_path = stub_dir / "stub_model.pkl"
    stub_metrics_path = stub_dir / "stub_metrics.json"

    joblib.dump(_StubModel(), stub_model_path)
    stub_metrics_path.write_text(json.dumps(
        {"mae": 0.4321, "rmse": 0.9876, "r2": 0.8765}
    ))

    # Only override env vars if the real model is not available. This keeps
    # the tests honest when the user has trained models in the repo root.
    if not Path(os.getenv("MODEL_PATH", "rf_tuned_model.pkl")).exists():
        os.environ["MODEL_PATH"] = str(stub_model_path)
        os.environ["METRICS_PATH"] = str(stub_metrics_path)

    # Ensure we don't try MLflow during tests (port 5001 isn't running in CI)
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    yield


@pytest.fixture(scope="module")
def client():
    """A TestClient context manager triggers the FastAPI lifespan events."""
    from app import app
    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────────
# Test 1 — Successful single prediction
# ──────────────────────────────────────────────────────────────────────────
def test_predict_single_success(client):
    """POST /predict with a valid trip returns a well-formed response."""
    response = client.post("/predict", json=VALID_TRIP)
    assert response.status_code == 200, response.text

    body = response.json()
    assert set(body.keys()) == {"prediction_id", "tip_amount", "model_version"}
    assert isinstance(body["tip_amount"], (int, float))
    assert isinstance(body["prediction_id"], str) and len(body["prediction_id"]) == 36
    # Value should be rounded to 2 decimals
    assert round(body["tip_amount"], 2) == body["tip_amount"]
    # Tip should be non-negative for a sensible fare
    assert body["tip_amount"] >= 0


# ──────────────────────────────────────────────────────────────────────────
# Test 2 — Successful batch prediction
# ──────────────────────────────────────────────────────────────────────────
def test_predict_batch_success(client):
    """POST /predict/batch with 3 trips returns 3 predictions, each with a UUID."""
    payload = {"trips": [VALID_TRIP, VALID_TRIP, VALID_TRIP]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3
    assert "batch_id" in body
    for p in body["predictions"]:
        assert set(p.keys()) == {"prediction_id", "tip_amount", "model_version"}
        assert isinstance(p["tip_amount"], (int, float))


def test_predict_batch_exceeds_limit(client):
    """Batches of more than 100 trips must be rejected with 422."""
    payload = {"trips": [VALID_TRIP] * 101}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────────────
# Test 3 — Invalid input rejection (multiple cases parameterized)
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("mutation,description", [
    ({"pickup_hour": 25},        "pickup_hour above 23"),
    ({"pickup_hour": -1},        "pickup_hour below 0"),
    ({"fare_amount": -5.0},      "negative fare_amount"),
    ({"fare_amount": 0},         "zero fare_amount (must be >0)"),
    ({"passenger_count": "two"}, "wrong type for passenger_count"),
    ({"pickup_day_of_week": 9},  "day out of range"),
])
def test_predict_rejects_invalid_input(client, mutation, description):
    """Any field violating its constraint must return HTTP 422."""
    bad_trip = {**VALID_TRIP, **mutation}
    response = client.post("/predict", json=bad_trip)
    assert response.status_code == 422, f"Should reject: {description} — got {response.status_code}"
    # Error body should describe which field failed
    detail = response.json().get("detail")
    assert detail is not None and len(detail) > 0


def test_predict_rejects_missing_field(client):
    """Missing a required field should return 422."""
    incomplete = {k: v for k, v in VALID_TRIP.items() if k != "fare_amount"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


# ──────────────────────────────────────────────────────────────────────────
# Test 4 — Health check endpoint
# ──────────────────────────────────────────────────────────────────────────
def test_health_endpoint(client):
    """GET /health returns status, model_loaded, and model_version."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert isinstance(body["model_version"], str)
    assert body["model_version"] != ""


def test_model_info_endpoint(client):
    """GET /model/info returns the expected metadata and metrics."""
    response = client.get("/model/info")
    assert response.status_code == 200

    body = response.json()
    assert body["model_name"]
    assert body["model_version"]
    assert body["feature_names"] == FEATURE_NAMES
    assert set(body["metrics"].keys()) == {"mae", "rmse", "r2"}


# ──────────────────────────────────────────────────────────────────────────
# Test 5 — Edge cases
# ──────────────────────────────────────────────────────────────────────────
def test_extreme_fare_value_accepted(client):
    """A very high but still-valid fare should produce a prediction without error."""
    extreme = {**VALID_TRIP, "fare_amount": 499.99, "trip_duration_minutes": 120.0}
    response = client.post("/predict", json=extreme)
    assert response.status_code == 200
    assert response.json()["tip_amount"] >= 0


def test_fare_over_cap_rejected(client):
    """fare_amount above 500 must be rejected."""
    over_cap = {**VALID_TRIP, "fare_amount": 501}
    response = client.post("/predict", json=over_cap)
    assert response.status_code == 422


def test_minimum_passenger_count(client):
    """A trip with zero passengers (data-entry edge case) is still accepted."""
    zero_pass = {**VALID_TRIP, "passenger_count": 0}
    response = client.post("/predict", json=zero_pass)
    assert response.status_code == 200


# ──────────────────────────────────────────────────────────────────────────
# Swagger docs
# ──────────────────────────────────────────────────────────────────────────
def test_swagger_docs_accessible(client):
    """/docs must be reachable so the auto-generated UI works."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower()


def test_openapi_schema_accessible(client):
    """The OpenAPI JSON schema backing Swagger must be valid JSON."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert "/predict/batch" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/model/info" in schema["paths"]
