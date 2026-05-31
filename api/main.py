# api/main.py
# FastAPI app that serves the fraud detection model as a REST API

import joblib
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------
# Load model and feature names once at startup
# -----------------------------------------------------------
MODEL_PATH    = Path("models/best_model.pkl")
FEATURES_PATH = Path("data/processed/feature_names.json")
DRIFT_PATH    = Path("monitoring/reports/latest_drift_summary.json")

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH) as f:
    FEATURES = json.load(f)

# -----------------------------------------------------------
# In-memory prediction log (last 100 predictions)
# -----------------------------------------------------------
prediction_log = deque(maxlen=100)

# -----------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud prediction using XGBoost — MLOps Internship Project",
    version="1.0.0"
)

# -----------------------------------------------------------
# Request schema — matches our 12 feature columns exactly
# -----------------------------------------------------------
class TransactionRequest(BaseModel):
    step: int                      = Field(..., example=1,         description="Hour of simulation")
    amount: float                  = Field(..., example=9839.64,   description="Transaction amount")
    oldbalanceOrg: float           = Field(..., example=170136.0,  description="Sender balance before")
    newbalanceOrig: float          = Field(..., example=160296.36, description="Sender balance after")
    oldbalanceDest: float          = Field(..., example=0.0,       description="Recipient balance before")
    newbalanceDest: float          = Field(..., example=0.0,       description="Recipient balance after")
    type_encoded: int              = Field(..., example=1,         description="1=TRANSFER, 0=CASH_OUT")
    orig_balance_diff: float       = Field(..., example=9839.64,   description="Sender balance difference")
    dest_balance_diff: float       = Field(..., example=0.0,       description="Recipient balance difference")
    orig_balance_zero: int         = Field(..., example=0,         description="1 if sender balance went to zero")
    amount_to_balance_ratio: float = Field(..., example=0.058,     description="Amount / sender original balance")
    hour_of_day: int               = Field(..., example=1,         description="Hour of day (step mod 24)")

# -----------------------------------------------------------
# Response schema
# -----------------------------------------------------------
class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str           # LOW / MEDIUM / HIGH
    prediction_time: str

# -----------------------------------------------------------
# Helper — reload model from disk into memory
# Called after retraining so the running API uses the new model
# -----------------------------------------------------------
def reload_model():
    global model
    model = joblib.load(MODEL_PATH)

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.get("/")
def root():
    return {
        "message":   "Fraud Detection API is running",
        "docs":      "/docs",
        "health":    "/health",
        "dashboard": "/dashboard"
    }


@app.get("/health")
def health():
    return {
        "status":    "healthy",
        "model":     MODEL_PATH.name,
        "features":  len(FEATURES),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    try:
        # Build single-row dataframe in exact feature order the model expects
        data = pd.DataFrame([transaction.model_dump()], columns=FEATURES)

        prediction  = model.predict(data)[0]
        probability = float(model.predict_proba(data)[0][1])

        # Risk level based on fraud probability
        if probability < 0.3:
            risk = "LOW"
        elif probability < 0.7:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        result = PredictionResponse(
            is_fraud          = bool(prediction),
            fraud_probability = round(probability, 4),
            risk_level        = risk,
            prediction_time   = datetime.now(timezone.utc).isoformat()
        )

        # Append to in-memory log for the dashboard
        prediction_log.append({
            "time":        result.prediction_time,
            "is_fraud":    result.is_fraud,
            "probability": result.fraud_probability,
            "risk":        result.risk_level
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
def model_info():
    return {
        "model_file":   MODEL_PATH.name,
        "features":     FEATURES,
        "num_features": len(FEATURES),
        "description":  "XGBoost trained on PaySim synthetic financial transactions"
    }


@app.get("/metrics")
def metrics():
    """JSON metrics endpoint — drift status + live prediction stats."""
    drift = {}
    if DRIFT_PATH.exists():
        with open(DRIFT_PATH) as f:
            drift = json.load(f)

    logs        = list(prediction_log)
    total       = len(logs)
    fraud_count = sum(1 for p in logs if p["is_fraud"])
    high_risk   = sum(1 for p in logs if p["risk"] == "HIGH")

    return {
        "predictions_served": total,
        "fraud_detected":     fraud_count,
        "fraud_rate":         round(fraud_count / total, 4) if total else 0,
        "high_risk_count":    high_risk,
        "drift_status":       drift,
        "model_file":         MODEL_PATH.name,
        "api_version":        "1.0.0"
    }


@app.post("/retrain")
def retrain(x_api_key: str = Header(default=None)):
    """
    Trigger drift detection and automated model retraining.
    Requires X-Api-Key header matching the RETRAIN_API_KEY environment variable.
    Default key for local dev: 'dev-secret'

    Example:
        curl -X POST http://localhost:8000/retrain -H "X-Api-Key: dev-secret"
    """
    # Simple API key check — prevents accidental or malicious retrain calls
    expected_key = os.getenv("RETRAIN_API_KEY", "dev-secret")
    if x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-Api-Key header"
        )

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from monitoring.drift_monitor import run_drift_and_retrain

        result = run_drift_and_retrain()

        # Reload the model into memory if retraining ran
        if result.get("retrain", {}).get("status") == "success":
            reload_model()

        return {
            "status":  "complete",
            "message": "Drift check and retrain pipeline finished.",
            "result":  result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {str(e)}")


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """Live monitoring dashboard — auto-refreshes every 30 seconds."""
    drift = {}
    if DRIFT_PATH.exists():
        with open(DRIFT_PATH) as f:
            drift = json.load(f)

    logs        = list(prediction_log)
    total       = len(logs)
    fraud_count = sum(1 for p in logs if p["is_fraud"])
    fraud_rate  = round(fraud_count / total * 100, 1) if total else 0
    high_risk   = sum(1 for p in logs if p["risk"] == "HIGH")

    drift_share   = drift.get("drift_share", 0)
    drift_pct     = round(drift_share * 100, 1)
    drifted_feats = drift.get("drifted_features", 0)
    total_feats   = drift.get("total_features", 12)
    retrain_status = "Triggered ✓" if drift.get("retrain_triggered") else "Not needed"
    drift_ts      = drift.get("timestamp", "N/A")

    # Build feature drift table rows
    feature_rows = ""
    for feat, drifted in drift.get("feature_drift", {}).items():
        badge = (
            '<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;'
            'border-radius:4px;font-size:12px">Drifted</span>'
            if drifted else
            '<span style="background:#dcfce7;color:#166534;padding:2px 8px;'
            'border-radius:4px;font-size:12px">Stable</span>'
        )
        feature_rows += (
            f"<tr><td style='padding:8px 12px'>{feat}</td>"
            f"<td style='padding:8px 12px'>{badge}</td></tr>"
        )

    if not feature_rows:
        feature_rows = (
            "<tr><td colspan='2' style='padding:12px;text-align:center;"
            "color:#6b7280'>No drift data yet — call POST /retrain to run drift detection</td></tr>"
        )

    # Build recent predictions table rows
    recent_rows = ""
    for p in reversed(list(prediction_log)[-10:]):
        color = "#991b1b" if p["is_fraud"] else "#166534"
        recent_rows += (
            f"<tr>"
            f"<td style='padding:6px 12px;font-size:12px'>{p['time'][:19]}</td>"
            f"<td style='padding:6px 12px;color:{color};font-weight:500'>"
            f"{'FRAUD' if p['is_fraud'] else 'LEGIT'}</td>"
            f"<td style='padding:6px 12px'>{p['probability']}</td>"
            f"<td style='padding:6px 12px'>{p['risk']}</td>"
            f"</tr>"
        )

    if not recent_rows:
        recent_rows = (
            "<tr><td colspan='4' style='padding:12px;text-align:center;color:#6b7280'>"
            "No predictions yet — call POST /predict to see data here</td></tr>"
        )

    # Drift bar color: green < 30%, amber 30-50%, red > 50%
    bar_color = (
        "#ef4444" if drift_pct > 50
        else "#f59e0b" if drift_pct > 30
        else "#22c55e"
    )
    drift_badge_class = "badge-bad" if drift.get("dataset_drifted") else "badge-ok"
    drift_badge_text  = "Dataset drifted" if drift.get("dataset_drifted") else "No drift detected"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    threshold_pct = drift.get("threshold", 0.3) * 100
    last_scan = drift_ts[:19] if drift_ts != "N/A" else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<title>Fraud Detection — Monitoring Dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f8fafc; color: #1e293b; padding: 24px; }}
  h1 {{ font-size: 22px; font-weight: 600; color: #0f172a; margin-bottom: 4px; }}
  .subtitle {{ font-size: 13px; color: #64748b; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
           gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; }}
  .card h3 {{ font-size: 12px; color: #64748b; text-transform: uppercase;
              letter-spacing: .05em; margin-bottom: 8px; }}
  .card .val {{ font-size: 32px; font-weight: 700; color: #0f172a; }}
  .card .sub {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
  .section {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
              padding: 20px; margin-bottom: 20px; }}
  .section h2 {{ font-size: 15px; font-weight: 600; margin-bottom: 14px; color: #0f172a; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th {{ text-align: left; padding: 8px 12px; font-size: 12px; color: #64748b;
        border-bottom: 1px solid #e2e8f0; font-weight: 500; }}
  td {{ border-bottom: 1px solid #f1f5f9; }}
  .drift-bar-wrap {{ background: #f1f5f9; border-radius: 6px; height: 10px; margin-top: 6px; }}
  .drift-bar {{ height: 10px; border-radius: 6px;
                background: {bar_color}; width: {drift_pct}%; }}
  .badge-ok  {{ background:#dcfce7;color:#166534;padding:3px 10px;border-radius:5px;font-size:12px; }}
  .badge-bad {{ background:#fee2e2;color:#991b1b;padding:3px 10px;border-radius:5px;font-size:12px; }}
  footer {{ font-size: 12px; color: #94a3b8; margin-top: 16px; text-align: center; }}
</style>
</head>
<body>
<h1>&#128737;&#65039; Fraud Detection &#8212; Monitoring Dashboard</h1>
<p class="subtitle">Auto-refreshes every 30 seconds &nbsp;&middot;&nbsp; Last updated: {now}</p>

<div class="grid">
  <div class="card">
    <h3>Predictions served</h3>
    <div class="val">{total}</div>
    <div class="sub">since last restart</div>
  </div>
  <div class="card">
    <h3>Fraud detected</h3>
    <div class="val" style="color:{'#dc2626' if fraud_count > 0 else '#16a34a'}">{fraud_count}</div>
    <div class="sub">{fraud_rate}% of total</div>
  </div>
  <div class="card">
    <h3>High-risk alerts</h3>
    <div class="val" style="color:{'#d97706' if high_risk > 0 else '#16a34a'}">{high_risk}</div>
    <div class="sub">probability &gt; 0.7</div>
  </div>
  <div class="card">
    <h3>Data drift share</h3>
    <div class="val">{drift_pct}%</div>
    <div class="drift-bar-wrap"><div class="drift-bar"></div></div>
    <div class="sub">{drifted_feats} / {total_feats} features drifted</div>
  </div>
</div>

<div class="section">
  <h2>Drift monitoring &nbsp;
    <span class="{drift_badge_class}">{drift_badge_text}</span>
    &nbsp;&nbsp;
    <span style="font-size:13px;color:#64748b">Retraining: {retrain_status}</span>
  </h2>
  <p style="font-size:13px;color:#64748b;margin-bottom:12px">
    Threshold: {threshold_pct:.0f}% &nbsp;&middot;&nbsp; Last scan: {last_scan}
  </p>
  <table>
    <thead><tr><th>Feature</th><th>Status</th></tr></thead>
    <tbody>{feature_rows}</tbody>
  </table>
</div>

<div class="section">
  <h2>Recent predictions (last 10)</h2>
  <table>
    <thead>
      <tr><th>Timestamp</th><th>Result</th><th>Probability</th><th>Risk level</th></tr>
    </thead>
    <tbody>{recent_rows}</tbody>
  </table>
</div>

<div class="section">
  <h2>System info</h2>
  <table>
    <tr>
      <td style='padding:8px 12px;color:#64748b'>Model file</td>
      <td style='padding:8px 12px'>{MODEL_PATH.name}</td>
    </tr>
    <tr>
      <td style='padding:8px 12px;color:#64748b'>Features</td>
      <td style='padding:8px 12px'>{len(FEATURES)} ({', '.join(FEATURES[:4])}…)</td>
    </tr>
    <tr>
      <td style='padding:8px 12px;color:#64748b'>API version</td>
      <td style='padding:8px 12px'>1.0.0</td>
    </tr>
    <tr>
      <td style='padding:8px 12px;color:#64748b'>Links</td>
      <td style='padding:8px 12px'>
        <a href="/docs" style="color:#3b82f6">/docs</a> &nbsp;&middot;&nbsp;
        <a href="/metrics" style="color:#3b82f6">/metrics</a>
      </td>
    </tr>
  </table>
</div>

<footer>Fraud Detection MLOps Platform &middot; Nisansala Ruwan Pathirana &middot; Internship 2026</footer>
</body>
</html>"""
    return HTMLResponse(content=html)