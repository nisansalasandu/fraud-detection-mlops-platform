# tests/test_train.py
# Model and data pipeline tests

import pytest
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path


def test_model_file_exists():
    """best_model.pkl must exist"""
    assert Path("models/best_model.pkl").exists()


def test_feature_names_file_exists():
    """feature_names.json must exist"""
    assert Path("data/processed/feature_names.json").exists()


def test_feature_names_count():
    """Must have exactly 12 features"""
    with open("data/processed/feature_names.json") as f:
        features = json.load(f)
    assert len(features) == 12


def test_model_can_predict():
    """Model must accept 12 features and return predictions"""
    model = joblib.load("models/best_model.pkl")
    sample = np.array([[1, 500.0, 5000.0, 4500.0, 1000.0,
                        1500.0, 0, 500.0, 500.0, 0, 0.1, 10]])
    pred = model.predict(sample)
    assert pred[0] in [0, 1]


def test_model_predict_proba():
    """Model must return probabilities between 0 and 1"""
    model = joblib.load("models/best_model.pkl")
    sample = np.array([[1, 500.0, 5000.0, 4500.0, 1000.0,
                        1500.0, 0, 500.0, 500.0, 0, 0.1, 10]])
    proba = model.predict_proba(sample)
    assert proba.shape[1] == 2
    assert 0.0 <= proba[0][1] <= 1.0


def test_model_output_type():
    """Model prediction must be integer 0 or 1"""
    model = joblib.load("models/best_model.pkl")
    sample = np.array([[1, 180000.0, 180000.0, 0.0, 0.0,
                        0.0, 1, 180000.0, 0.0, 1, 1.0, 1]])
    pred = model.predict(sample)
    assert int(pred[0]) in [0, 1]