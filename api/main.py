# api/main.py
# FastAPI app that serves the fraud detection model as a REST API

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# -----------------------------------------------------------
# Load model and feature names once at startup
# -----------------------------------------------------------
MODEL_PATH   = Path("models/best_model.pkl")
FEATURES_PATH = Path("data/processed/feature_names.json")

model = joblib.load(MODEL_PATH)

with open(FEATURES_PATH) as f:
    FEATURES = json.load(f)

# -----------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud prediction using XGBoost — MLOps Project",
    version="1.0.0"
)

# -----------------------------------------------------------
# Request schema — matches our 12 feature columns exactly
# -----------------------------------------------------------
class TransactionRequest(BaseModel):
    step: int                        = Field(..., example=1,        description="Hour of simulation")
    amount: float                    = Field(..., example=9839.64,   description="Transaction amount")
    oldbalanceOrg: float             = Field(..., example=170136.0,  description="Sender balance before")
    newbalanceOrig: float            = Field(..., example=160296.36, description="Sender balance after")
    oldbalanceDest: float            = Field(..., example=0.0,       description="Recipient balance before")
    newbalanceDest: float            = Field(..., example=0.0,       description="Recipient balance after")
    type_encoded: int                = Field(..., example=1,         description="1=TRANSFER, 0=CASH_OUT")
    orig_balance_diff: float         = Field(..., example=9839.64,   description="Sender balance difference")
    dest_balance_diff: float         = Field(..., example=0.0,       description="Recipient balance difference")
    orig_balance_zero: int           = Field(..., example=0,         description="1 if sender balance went to zero")
    amount_to_balance_ratio: float   = Field(..., example=0.058,     description="Amount / sender original balance")
    hour_of_day: int                 = Field(..., example=1,         description="Hour of day (step mod 24)")

# -----------------------------------------------------------
# Response schema
# -----------------------------------------------------------
class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_level: str          # LOW / MEDIUM / HIGH
    prediction_time: str

# -----------------------------------------------------------
# Routes
# -----------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "docs":    "/docs",
        "health":  "/health"
    }

@app.get("/health")
def health():
    return {
        "status":     "healthy",
        "model":      MODEL_PATH.name,
        "features":   len(FEATURES),
        "timestamp":  datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    try:
        # Build a single-row dataframe in the exact feature order
        data = pd.DataFrame([transaction.model_dump()], columns=FEATURES)

        # Get prediction and probability
        prediction  = model.predict(data)[0]
        probability = float(model.predict_proba(data)[0][1])

        # Assign risk level based on probability
        if probability < 0.3:
            risk = "LOW"
        elif probability < 0.7:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return PredictionResponse(
            is_fraud          = bool(prediction),
            fraud_probability = round(probability, 4),
            risk_level        = risk,
            prediction_time   = datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_file":  MODEL_PATH.name,
        "features":    FEATURES,
        "num_features": len(FEATURES),
        "description": "XGBoost trained on PaySim synthetic financial transactions"
    }