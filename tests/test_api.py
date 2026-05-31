# tests/test_api.py
# API endpoint tests — run by GitHub Actions on every push

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Sample transactions for testing
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
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0,
    "type_encoded": 1,
    "orig_balance_diff": 180000.0,
    "dest_balance_diff": 0.0,
    "orig_balance_zero": 1,
    "amount_to_balance_ratio": 1.0,
    "hour_of_day": 1
}


def test_root_endpoint():
    """GET / should return 200 with a message"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_health_endpoint():
    """GET /health should return healthy status"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["features"] == 12


def test_model_info_endpoint():
    """GET /model-info should return feature list"""
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_features"] == 12
    assert "features" in data


def test_predict_normal_transaction():
    """POST /predict should return valid response for normal transaction"""
    resp = client.post("/predict", json=NORMAL_TRANSACTION)
    assert resp.status_code == 200
    data = resp.json()
    assert "is_fraud" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_suspicious_transaction():
    """POST /predict should return valid response for suspicious transaction"""
    resp = client.post("/predict", json=SUSPICIOUS_TRANSACTION)
    assert resp.status_code == 200
    data = resp.json()
    assert "is_fraud" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_missing_field():
    """POST /predict with missing field should return 422"""
    incomplete = {"step": 1, "amount": 500.0}
    resp = client.post("/predict", json=incomplete)
    assert resp.status_code == 422


def test_predict_response_schema():
    """POST /predict response must have all required fields"""
    resp = client.post("/predict", json=NORMAL_TRANSACTION)
    data = resp.json()
    required_fields = ["is_fraud", "fraud_probability", "risk_level", "prediction_time"]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"