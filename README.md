# 🛡️ Fraud Detection MLOps Platform

> Real-time AI prediction platform with automated retraining, drift monitoring, and CI/CD.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.12-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Railway](https://img.shields.io/badge/Railway-deployed-purple)
![CI/CD](https://github.com/nisansalasandu/fraud-detection-mlops-platform/actions/workflows/ci_cd.yml/badge.svg)

## 🌐 Live API
| Endpoint | URL |
|---|---|
| Base URL | https://fraud-detection-mlops-platform-production-production.up.railway.app |
| Interactive Docs | https://fraud-detection-mlops-platform-production-production.up.railway.app/docs |
| Health Check | https://fraud-detection-mlops-platform-production-production.up.railway.app/health |

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
│                    DATA LAYER                                │
│  PaySim Dataset (6.3M rows) → Feature Engineering → SMOTE  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  TRAINING LAYER                              │
│  XGBoost Model ←→ MLflow Experiment Tracking                │
│  Logistic Regression (baseline)  Model Registry             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  SERVING LAYER                               │
│  FastAPI  →  POST /predict  →  JSON Response                │
│  Pydantic validation  |  Risk scoring (LOW/MEDIUM/HIGH)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                MONITORING LAYER                              │
│  Evidently AI → Drift Reports → Auto-Retrain if >30% drift  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│               DEPLOYMENT LAYER                               │
│  Docker Image → Railway Cloud → Live Public URL             │
│  GitHub Actions CI/CD → pytest → Auto Deploy on push        │
└─────────────────────────────────────────────────────────────┘
```


## 🚀 Quick Start
```bash
git clone https://github.com/nisansalasandu/fraud-detection-mlops-platform.git
cd fraud-detection-mlops-platform
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload
# → http://localhost:8000/docs
```

## 🐳 Run with Docker
```bash
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api
```

## 📊 Model Performance
| Model | ROC-AUC | F1 Score | Recall |
|---|---|---|---|
| Logistic Regression (baseline) | 0.9960 | 0.1915 | 0.9832 |
| XGBoost (final) | 0.9960 | 0.7945 | 0.7227 |

## 🔁 MLOps Pipeline
1. **Data** — PaySim synthetic financial dataset (6.3M transactions)
2. **Preprocessing** — Feature engineering, SMOTE for class imbalance
3. **Training** — XGBoost with MLflow experiment tracking
4. **Serving** — FastAPI REST API with Pydantic validation
5. **Monitoring** — Evidently data drift detection
6. **Retraining** — Auto-triggered when drift exceeds 30% threshold
7. **Deployment** — Docker + Railway with GitHub Actions CI/CD

## 📦 Tech Stack
| Layer | Technology |
|---|---|
| ML | scikit-learn, XGBoost, imbalanced-learn |
| Experiment Tracking | MLflow |
| Drift Monitoring | Evidently AI |
| API | FastAPI + Uvicorn |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Railway |
| Language | Python 3.12 |

## 📁 Project Structure

```

fraud-detection-mlops-platform/
├── api/
│   └── main.py              # FastAPI app
├── notebooks/
│   ├── 01_data_download_and_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_api_and_testing.ipynb
│   └── 04_drift_monitoring.ipynb
├── models/
│   └── best_model.pkl       # Trained XGBoost model
├── monitoring/
│   └── reports/             # Evidently HTML + JSON reports
├── tests/
│   ├── test_api.py
│   └── test_train.py
├── data/processed/
│   └── feature_names.json
├── Dockerfile
├── docker-compose.yml
├── railway.toml
├── requirements.txt
└── .github/workflows/
└── ci_cd.yml

```

## 🔌 API Usage

### Predict endpoint
```bash
curl -X POST "https://fraud-detection-mlops-platform-production-production.up.railway.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "amount": 180000.0,
    "oldbalanceOrg": 180000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "type_encoded": 1,
    "orig_balance_diff": 180000.0,
    "dest_balance_diff": 0.0,
    "orig_balance_zero": 1,
    "amount_to_balance_ratio": 1.0,
    "hour_of_day": 1
  }'
```

### Response
```json
{
  "is_fraud": true,
  "fraud_probability": 0.9986,
  "risk_level": "HIGH",
  "prediction_time": "2026-05-31T13:00:00"
}
```

## 📓 Notebooks
| Notebook | Description |
|---|---|
| 01_data_download_and_preprocessing | EDA, feature engineering, train/test split |
| 02_model_training | MLflow tracking, XGBoost training, model registry |
| 03_api_and_testing | FastAPI testing, batch prediction verification |
| 04_drift_monitoring | Evidently reports, auto-retrain trigger |

---
*Internship Project — Data Science Internship, May–June 2026*

*Nisansala Ruwan Pathirana*
