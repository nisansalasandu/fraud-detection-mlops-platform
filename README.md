# 🛡️ Fraud Detection MLOps Platform

> Real-time AI prediction platform with automated retraining, drift monitoring, and CI/CD.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.12-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Railway](https://img.shields.io/badge/Railway-deployed-purple)
![CI/CD](https://github.com/nisansalasandu/fraud-detection-mlops-platform/actions/workflows/ci_cd.yml/badge.svg)

## 🌐 Live API
|Endpoint|URL|
|-|-|
|Base URL|https://fraud-detection-mlops-platform-production-production.up.railway.app|
|Interactive Docs|https://fraud-detection-mlops-platform-production-production.up.railway.app/docs|
|Health Check|https://fraud-detection-mlops-platform-production-production.up.railway.app/health|
|**Monitoring Dashboard**|https://fraud-detection-mlops-platform-production-production.up.railway.app/dashboard|

\---


## 🏗️ Architecture

```
PaySim Dataset → Preprocessing → MLflow Training → Model Registry
↓
XGBoost Model → FastAPI → Docker → Railway (Live)
↓
Evidently Drift Monitor → Auto-Retrain Trigger → Updated Model
↓
GitHub Actions CI/CD → Automated Tests → Auto Deploy

```

# Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  PaySim Dataset (6.3M rows)                                  │
│  → Feature Engineering (12 features, balance diffs, ratios) │
│  → SMOTE oversampling for class imbalance                    │
│  → Train/test split saved to data/processed/                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    TRAINING LAYER                            │
│  Logistic Regression (baseline) ←→ MLflow Experiment #1     │
│  XGBoost (final model)          ←→ MLflow Experiment #2     │
│  Best model saved to models/best\_model.pkl                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    SERVING LAYER                             │
│  FastAPI REST API (Uvicorn)                                  │
│  POST /predict  → JSON { is\_fraud, probability, risk\_level } │
│  GET  /health   → service status                             │
│  GET  /metrics  → JSON metrics snapshot                      │
│  GET  /dashboard → live HTML monitoring dashboard            │
│  POST /retrain  → trigger drift check + retraining           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  MONITORING LAYER                            │
│  Evidently AI → DataDriftPreset → per-feature drift report   │
│  drift\_share > 30% threshold → automated XGBoost retrain    │
│  New model logged to MLflow, saved to models/best\_model.pkl  │
│  Drift summary written to monitoring/reports/                │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  DEPLOYMENT LAYER                            │
│  Docker Image (python:3.12-slim) → Railway Cloud (live URL)  │
│  GitHub Actions CI/CD:                                       │
│    1. Checkout code                                          │
│    2. Install dependencies                                   │
│    3. Create test fixtures (dummy model for CI)              │
│    4. Run pytest (API + model tests)                         │
│    5. Build Docker image                                     │
│  Push to main → auto deploy on Railway                      │
└─────────────────────────────────────────────────────────────┘
```

\---


## 🚀 Quick Start

```bash
git clone https://github.com/nisansalasandu/fraud-detection-mlops-platform.git
cd fraud-detection-mlops-platform

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\\Scripts\\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt

# Generate model (required — .pkl files are excluded from git)
jupyter notebook notebooks/02\_model\_training.ipynb

# Start the API
uvicorn api.main:app --reload
# → http://localhost:8000/docs        (interactive API docs)
# → http://localhost:8000/dashboard   (monitoring dashboard)
```


## 🐳 Run with Docker
```bash
# Build and run
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api

# Or use docker-compose (mounts models/ so retrained models take effect without rebuild)
docker-compose up
```

\---

## 📊 Model Performance

|Model|ROC-AUC|F1 Score|Recall|Notes|
|-|-|-|-|-|
|Logistic Regression (baseline)|0.9960|0.1915|0.9832|High recall, low precision|
|**XGBoost (final)**|**0.9960**|**0.7945**|0.7227|Best balance of precision/recall|

\---

## 🔁 MLOps Pipeline — Full Flow

### 1\. Data

PaySim synthetic financial transactions dataset (6.3M rows). Downloaded via Kaggle API in `01\_data\_download\_and\_preprocessing.ipynb`. Features engineered: balance diffs, zero-balance flags, amount/balance ratio, hour of day.

### 2\. Training

`02\_model\_training.ipynb` trains Logistic Regression (baseline) and XGBoost, tracking both runs in MLflow. Best model saved to `models/best\_model.pkl`.

### 3\. Serving

`api/main.py` loads the model at startup and serves predictions via FastAPI:

* `POST /predict` — real-time fraud prediction with risk scoring
* `GET /health` — service health check
* `GET /dashboard` — live HTML monitoring dashboard (auto-refreshes every 30s)
* `GET /metrics` — JSON metrics for external monitoring
* `POST /retrain` — trigger full drift check + retraining pipeline on demand

### 4\. Drift Monitoring \& Automated Retraining

`monitoring/drift\_monitor.py` implements the full automated pipeline:

1. Runs `Evidently DataDriftPreset` between reference (training) and current (incoming) data
2. Calculates `drift\_share` — fraction of features that have drifted
3. If `drift\_share > 0.30` (30% threshold) → automatically retrains XGBoost on combined data
4. Logs the retraining run (params + metrics) to MLflow
5. Saves the updated model to `models/best\_model.pkl`
6. Writes drift summary to `monitoring/reports/latest\_drift\_summary.json`

To trigger manually:

```bash
python -m monitoring.drift\_monitor
```

To trigger via API:

```bash
curl -X POST http://localhost:8000/retrain
```

In a production environment, this pipeline would be scheduled using APScheduler or a cron job:

```python
# Example: run drift check every 6 hours
from apscheduler.schedulers.background import BackgroundScheduler
from monitoring.drift\_monitor import run\_drift\_and\_retrain

scheduler = BackgroundScheduler()
scheduler.add\_job(run\_drift\_and\_retrain, "interval", hours=6)
scheduler.start()
```

### 5\. CI/CD

`.github/workflows/ci\_cd.yml` runs on every push to `main`:

1. Creates dummy model fixtures (so CI doesn't need the large `.pkl` files)
2. Runs `pytest tests/` — 9 tests covering API endpoints and model validation
3. Builds Docker image to confirm `Dockerfile` is valid
4. Railway auto-deploys on successful push to `main`

\---

## 📦 Tech Stack

|Layer|Technology|
|-|-|
|ML|scikit-learn, XGBoost, imbalanced-learn (SMOTE)|
|Experiment Tracking|MLflow|
|Drift Monitoring|Evidently AI|
|API|FastAPI + Uvicorn|
|Containerisation|Docker + docker-compose|
|CI/CD|GitHub Actions|
|Deployment|Railway|
|Language|Python 3.12|

\---

## 📁 Project Structure

```
fraud-detection-mlops-platform/
├── api/
│   └── main.py                  # FastAPI app (predict, dashboard, retrain)
├── monitoring/
│   ├── drift\_monitor.py         # Drift detection + automated retraining module
│   └── reports/
│       ├── latest\_drift\_summary.json
│       └── drift\_report\_\*.html  # Evidently HTML reports (git-ignored)
├── notebooks/
│   ├── 01\_data\_download\_and\_preprocessing.ipynb
│   ├── 02\_model\_training.ipynb
│   ├── 03\_api\_and\_testing.ipynb
│   └── 04\_drift\_monitoring.ipynb
├── models/
│   └── best\_model.pkl           # Trained XGBoost (git-ignored, regenerate via nb 02)
├── tests/
│   ├── test\_api.py              # FastAPI endpoint tests
│   └── test\_train.py            # Model loading + prediction tests
├── data/processed/
│   └── feature\_names.json
├── .github/workflows/
│   └── ci\_cd.yml                # GitHub Actions CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── railway.json
├── requirements.txt
└── README.md
```

\---

## 🔌 API Usage

### Predict endpoint

```bash
curl -X POST "https://fraud-detection-mlops-platform-production-production.up.railway.app/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "step": 1,
    "amount": 180000.0,
    "oldbalanceOrg": 180000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "type\_encoded": 1,
    "orig\_balance\_diff": 180000.0,
    "dest\_balance\_diff": 0.0,
    "orig\_balance\_zero": 1,
    "amount\_to\_balance\_ratio": 1.0,
    "hour\_of\_day": 1
  }'
```

### Response

```json
{
  "is\_fraud": true,
  "fraud\_probability": 0.9986,
  "risk\_level": "HIGH",
  "prediction\_time": "2026-05-31T13:00:00+00:00"
}
```

### Trigger retraining

```bash
curl -X POST "https://fraud-detection-mlops-platform-production-production.up.railway.app/retrain"
```

\---

## 📓 Notebooks

|Notebook|Description|
|-|-|
|01\_data\_download\_and\_preprocessing|EDA, feature engineering, train/test split|
|02\_model\_training|MLflow tracking, XGBoost training, model registry|
|03\_api\_and\_testing|FastAPI testing, batch prediction verification|
|04\_drift\_monitoring|Evidently reports, auto-retrain trigger demo|

\---

*Internship Project — Data Science Internship, May–June 2026*

*Nisansala Ruwan Pathirana*
