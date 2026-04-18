# MLOps-Model-Deployment

Taxi Tip Prediction Service
COMP 3610: Big Data Analytics — Assignment 4 (MLOps & Model Deployment)
University of the West Indies · Semester II, 2025–2026
A containerised REST API that predicts the tip amount for a NYC Yellow Taxi trip, built on top of a Random Forest regressor trained in Assignment 2. Experiment runs are tracked with MLflow, the API is served with FastAPI, and the full stack runs via a single docker compose up command.

Project Overview
The service wraps a trained regression model (taxi-tip-regressor, Random Forest) and exposes it as a JSON HTTP API. A single prediction returns a tip estimate, a model version tag, and a UUID trace ID. Input validation is enforced with Pydantic, tests are written in pytest with FastAPI's TestClient, and the whole stack (API + MLflow tracking server) is orchestrated with Docker Compose.
Stack:


| Tool | Version | Purpose |
|---|---|---|
| Docker Desktop | 4.x or later | Running the Compose stack |
| Python | 3.11+ | Local development and tests (optional if using Docker) |
| Git | any recent | Cloning the repository |

Verify your install:

```bash
docker --version
docker compose version
python3 --version
```

---

## Quick Start (Docker)

This is the fastest path to a running prediction service.

```bash
# 1. Clone and enter the repo
git clone <your-repo-url> assignment4
cd assignment4

# 2. Start both services (API + MLflow tracking server)
docker compose up -d

# 3. Wait for startup (model load takes ~40s)
sleep 20
curl http://localhost:8000/health
# Expected: {"status":"ok","model_loaded":true,"model_version":"local-<timestamp>"}

# 4. Open the interactive docs
#    API Swagger UI: http://localhost:8000/docs
#    MLflow UI:      http://localhost:5000

# 5. Shut down cleanly when done
docker compose down
```

---

## Local Development (without Docker)

Use this path if you want to iterate on `app.py` or run tests without rebuilding the image.

```bash
# 1. Create and activate a virtual environment
cd assignment4
python3 -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate             # Windows PowerShell

# 2. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Run the API with auto-reload
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 4. In a second terminal, run the tests
pytest test_app.py -v
```

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Predict the tip for one trip |
| `POST` | `/predict/batch` | Predict tips for up to 100 trips in one call |
| `GET` | `/health` | Service and model readiness |
| `GET` | `/model/info` | Model name, version, features, training metrics |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `GET` | `/openapi.json` | Machine-readable OpenAPI schema |

**Health check:**

```bash
curl http://localhost:8000/health
```

---

## Testing

The suite contains 17 tests covering:

- Successful single prediction
- Successful batch prediction
- Batch limit enforcement (rejects >100 records)
- Six parametrised invalid-input cases (out-of-range hours, negative fare, zero fare, wrong type, bad day)
- Missing required field
- `/health` endpoint returns expected fields
- `/model/info` endpoint returns model metadata
- Edge case: extreme but valid fare accepted
- Edge case: fare above reasonable cap rejected
- Minimum passenger count accepted
- Swagger `/docs` accessibility
- OpenAPI JSON schema accessibility

**Run all tests:**

```bash
# In the activated venv
pytest test_app.py -v

# Expected summary:
# ================ 17 passed in 0.80s ================
```

---

## MLflow Tracking

When the Compose stack is up, the MLflow UI is available at **http://localhost:5000**.

- **Experiment:** `taxi-tip-prediction`
- **Runs logged:** Random Forest baseline + tuned Random Forest (and/or PyTorch regressor from Assignment 2)
- **Logged per run:** hyperparameters, MAE, RMSE, R², trained model artefact, tags (`model_type`, `dataset_version`)
- **Registry:** the best model is registered as `taxi-tip-regressor` with a version description summarising its metrics

To re-run the experiments inside the notebook:

```bash
jupyter lab assignment4.ipynb
# Run all cells under "Part 1: Experiment Tracking with MLflow"
```

---

## Environment Variables

Configurable via `docker-compose.yml` or a local `.env` file:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `/app/rf_model.pkl` | Where the API loads the model from at startup |
| `MLFLOW_TRACKING_URI` | `http://mlflow-server:5000` | MLflow server URL (service DNS inside the Compose network) |
| `API_PORT` | `8000` | Host port mapping for the FastAPI service |
| `LOG_LEVEL` | `INFO` | Uvicorn log level |

---

## Image & Container Info

| Item | Value |
|---|---|
| Base image | `python:3.11-slim` |
| Final image tag | `taxi-tip-api:latest` |
| First build time | ~70 seconds (uncached) |
| Rebuild time | ~3 seconds (cached layers) |
| Approximate image size | **[Fill in from `docker images taxi-tip-api:latest`]** |
| Compose network | `taxi-tip-network` |
| Compose volume | `mlflow-data` (persists MLflow runs between restarts) |

Check image size yourself:

```bash
docker images taxi-tip-api:latest --format "{{.Repository}}:{{.Tag}}  {{.Size}}"
```

---

## AI Tools Used

In line with the course's academic-integrity policy, the following AI tools were used during this assignment:

- **Claude (Anthropic)**
  - Drafted initial skeletons for the Pydantic request models and the `pytest` parametrised invalid-input tests in `test_app.py`.
  - Helped debug the `uvicorn` startup error `Error loading ASGI app. Attribute "app" not found in module "test_app"` (the fix was to pass `app:app`, not `test_app:app`, as the module path).
  - Organised the evidence report and drafted this README.

- **GitHub Copilot** *(remove if not used)*
  - Autocompletion while writing repetitive boilerplate (imports, pytest fixture patterns).

**Scope and verification.** All AI-generated code was read, edited, and tested locally before being committed. The model training pipeline, feature engineering, and the Random Forest regressor itself were carried over from Assignment 2 and are my own work. No AI tool was given the assignment rubric and asked to write the submission end-to-end. I am able to explain every file in this repository and the design choices behind it.

---

## Academic Integrity

This submission is my own work, produced in accordance with the UWI academic-integrity policy and the AI-usage guidance in the course outline. All external assistance (including AI tools, as disclosed above) has been acknowledged.

---

AI Tools Used
In line with the course's academic-integrity policy, the following AI tools were used during this assignment:

Claude (Anthropic)

Commenting Code
Troubleshooting & Debugging
Organised the evidence report and drafted this README.
