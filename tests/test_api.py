# tests/test_api.py
# API endpoint tests — run by GitHub Actions on every push

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# -----------------------------------------------------------
# Sample transactions used across multiple tests
# -----------------------------------------------------------
NORMAL_TRANSACTION = {
    "step": 10,
    "amount": 500.0,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4500.0,
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 1500.0,
    "type_encoded": 0,
    "orig_balance_diff": 500.0,
    "dest_balance_diff": 500.0,
    "orig_balance_zero": 0,
    "amount_to_balance_ratio": 0.1,
    "hour_of_day": 10
}

SUSPICIOUS_TRANSACTION = {
    "step": 1,
    "amount": 180000.0,
    "oldbalanceOrg": 180000.0,
    "newbalanceOrig": 0.0,       # balance wiped out
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,       # recipient doesn't receive funds
    "type_encoded": 1,           # TRANSFER
    "orig_balance_diff": 180000.0,
    "dest_balance_diff": 0.0,
    "orig_balance_zero": 1,      # sender went to zero
    "amount_to_balance_ratio": 1.0,
    "hour_of_day": 1
}


# -----------------------------------------------------------
# Root & health
# -----------------------------------------------------------
def test_root_endpoint():
    """GET / should return 200 with a message"""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data
    assert "dashboard" in data   # dashboard link added in main.py


def test_health_endpoint():
    """GET /health should return healthy status with correct feature count"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["features"] == 12
    assert "timestamp" in data
    assert "model" in data


# -----------------------------------------------------------
# Model info
# -----------------------------------------------------------
def test_model_info_endpoint():
    """GET /model-info should return feature list and count"""
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_features"] == 12
    assert "features" in data
    assert len(data["features"]) == 12


# -----------------------------------------------------------
# Predict
# -----------------------------------------------------------
def test_predict_normal_transaction():
    """POST /predict on a normal transaction should return valid response"""
    resp = client.post("/predict", json=NORMAL_TRANSACTION)
    assert resp.status_code == 200
    data = resp.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "prediction_time" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_suspicious_transaction():
    """POST /predict on a suspicious transaction should return valid response"""
    resp = client.post("/predict", json=SUSPICIOUS_TRANSACTION)
    assert resp.status_code == 200
    data = resp.json()
    assert "is_fraud" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_missing_field_returns_422():
    """POST /predict with missing required fields should return 422 Unprocessable Entity"""
    incomplete = {"step": 1, "amount": 500.0}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_response_has_all_fields():
    """POST /predict response must contain all four required fields"""
    resp = client.post("/predict", json=NORMAL_TRANSACTION)
    assert resp.status_code == 200
    data = resp.json()
    for field in ["is_fraud", "fraud_probability", "risk_level", "prediction_time"]:
        assert field in data, f"Missing field in response: {field}"


def test_predict_probability_range():
    """fraud_probability must always be between 0.0 and 1.0"""
    for transaction in [NORMAL_TRANSACTION, SUSPICIOUS_TRANSACTION]:
        resp = client.post("/predict", json=transaction)
        assert resp.status_code == 200
        prob = resp.json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"


def test_predict_is_fraud_is_boolean():
    """is_fraud field must be a boolean"""
    resp = client.post("/predict", json=NORMAL_TRANSACTION)
    assert resp.status_code == 200
    assert isinstance(resp.json()["is_fraud"], bool)


# -----------------------------------------------------------
# Metrics endpoint
# -----------------------------------------------------------
def test_metrics_endpoint():
    """GET /metrics should return JSON with all expected keys"""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions_served" in data
    assert "fraud_detected" in data
    assert "fraud_rate" in data
    assert "high_risk_count" in data
    assert "model_file" in data
    assert "api_version" in data


def test_metrics_fraud_rate_is_valid():
    """fraud_rate in /metrics should be between 0 and 1"""
    resp = client.get("/metrics")
    assert resp.status_code == 200
    rate = resp.json()["fraud_rate"]
    assert 0.0 <= rate <= 1.0


def test_metrics_updates_after_predict():
    """After calling /predict, predictions_served in /metrics should increase"""
    before = client.get("/metrics").json()["predictions_served"]
    client.post("/predict", json=NORMAL_TRANSACTION)
    after  = client.get("/metrics").json()["predictions_served"]
    assert after == before + 1


# -----------------------------------------------------------
# Dashboard endpoint
# -----------------------------------------------------------
def test_dashboard_returns_html():
    """GET /dashboard should return 200 with HTML content type"""
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_dashboard_contains_key_elements():
    """GET /dashboard HTML should contain the page title and key section headings"""
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    html = resp.text
    assert "Fraud Detection" in html
    assert "Drift monitoring" in html
    assert "Recent predictions" in html
    assert "System info" in html


# -----------------------------------------------------------
# Retrain endpoint — auth check only (no actual retrain in CI)
# -----------------------------------------------------------
def test_retrain_requires_api_key():
    """POST /retrain without X-Api-Key header should return 401"""
    resp = client.post("/retrain")
    assert resp.status_code == 401


def test_retrain_wrong_api_key_returns_401():
    """POST /retrain with wrong key should return 401"""
    resp = client.post("/retrain", headers={"X-Api-Key": "wrong-key"})
    assert resp.status_code == 401