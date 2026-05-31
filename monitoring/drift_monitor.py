# monitoring/drift_monitor.py
# Drift detection and automated retraining trigger.
# Used by:
#   - Notebook 04_drift_monitoring.ipynb (manual / scheduled runs)
#   - POST /retrain API endpoint (on-demand trigger)
#
# How it works:
#   1. Load reference data (training set) and current data (test set or new batch)
#   2. Run Evidently DataDriftPreset to compute per-feature drift
#   3. Save a JSON summary + HTML report to monitoring/reports/
#   4. If drift_share > threshold → retrain the model with combined data
#   5. Save updated model to models/best_model.pkl
#   6. Log the retraining run to MLflow

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost          # fixed: was mlflow.sklearn for an XGBoost model
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
REPORTS_DIR   = BASE_DIR / "monitoring" / "reports"
MODELS_DIR    = BASE_DIR / "models"
DATA_DIR      = BASE_DIR / "data" / "processed"
DRIFT_SUMMARY = REPORTS_DIR / "latest_drift_summary.json"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_THRESHOLD = 0.3   # retrain if more than 30% of features drift


# -----------------------------------------------------------
# Drift detection
# -----------------------------------------------------------
def detect_drift(
    reference_path = DATA_DIR / "X_train.csv",
    current_path   = DATA_DIR / "X_test.csv",
    threshold: float = DRIFT_THRESHOLD,
) -> dict:
    """
    Run Evidently drift detection between reference and current datasets.
    Returns a summary dict and writes JSON + HTML reports to monitoring/reports/.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        log.error("evidently is not installed. Run: pip install evidently==0.4.30")
        raise

    log.info("Loading reference data from %s", reference_path)
    reference = pd.read_csv(reference_path)
    current   = pd.read_csv(current_path)

    log.info(
        "Running Evidently DataDriftPreset (%d ref rows, %d cur rows)",
        len(reference), len(current)
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    result = report.as_dict()

    # -----------------------------------------------------------
    # Parse per-feature drift results
    # Evidently v0.4.x stores overall summary in metrics[0]
    # and per-feature results in metrics[1..N] each with "column_name"
    # -----------------------------------------------------------
    feature_drift: dict = {}
    try:
        metrics_list = result["metrics"]

        # Approach: iterate all metric entries, pick ones that have column_name
        for metric_entry in metrics_list:
            entry_result = metric_entry.get("result", {})
            if "column_name" in entry_result:
                col     = entry_result["column_name"]
                drifted = bool(entry_result.get("drift_detected", False))
                feature_drift[col] = drifted

        # Fallback: some versions use drift_by_columns in the first metric
        if not feature_drift:
            first = metrics_list[0].get("result", {})
            drift_by_columns = first.get("drift_by_columns", {})
            for col, info in drift_by_columns.items():
                feature_drift[col] = bool(info.get("drift_detected", False))

    except (KeyError, TypeError, IndexError) as e:
        log.warning("Could not parse per-feature drift details: %s", e)

    drifted      = sum(1 for v in feature_drift.values() if v)
    total_feats  = len(feature_drift) or 1
    drift_share  = round(drifted / total_feats, 4)
    drifted_flag = drift_share > threshold

    summary = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "dataset_drifted":   drifted_flag,
        "drifted_features":  drifted,
        "total_features":    total_feats,
        "drift_share":       drift_share,
        "threshold":         threshold,
        "retrain_triggered": drifted_flag,
        "feature_drift":     feature_drift,
    }

    # Save JSON summary — read by the /dashboard and /metrics endpoints
    with open(DRIFT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Drift summary saved to %s", DRIFT_SUMMARY)

    # Save full HTML report for manual review
    ts        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"drift_report_{ts}.html"
    report.save_html(str(html_path))
    log.info("Drift HTML report saved to %s", html_path)

    return summary


# -----------------------------------------------------------
# Automated retraining
# -----------------------------------------------------------
def retrain_model(summary: dict = None) -> dict:
    """
    Retrain the XGBoost model on combined train+test data.
    Logs the run to MLflow and saves the updated model to models/best_model.pkl.
    Returns a result dict with new metrics.
    """
    from sklearn.metrics import roc_auc_score, f1_score
    from xgboost import XGBClassifier

    log.info("Loading data for retraining...")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    X_test  = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    # Combine train + test to include all available data
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True)

    fraud_count     = int((y_combined == 1).sum())
    non_fraud_count = int((y_combined == 0).sum())

    params = {
        "n_estimators":     300,
        "max_depth":        6,
        "learning_rate":    0.05,
        "scale_pos_weight": non_fraud_count // max(fraud_count, 1),
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     42,
        "n_jobs":           -1,
        "eval_metric":      "logloss",
    }

    log.info("Retraining XGBoost on %d samples...", len(X_combined))
    clf = XGBClassifier(**params)
    clf.fit(X_combined, y_combined, verbose=False)

    # Evaluate on the held-out test set
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    roc_auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    f1      = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

    log.info("Retrained model — ROC-AUC: %.4f  F1: %.4f", roc_auc, f1)

    # Save model to disk
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(clf, model_path)
    log.info("Updated model saved to %s", model_path)

    # Log run to MLflow
    mlflow.set_tracking_uri(str(BASE_DIR / "mlruns"))
    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run(run_name="auto_retrain"):
        mlflow.log_params(params)
        mlflow.log_params({
            "trigger":       "drift_monitor",
            "drift_share":   summary.get("drift_share", "N/A") if summary else "manual",
            "drifted_feats": summary.get("drifted_features", "N/A") if summary else "manual",
        })
        mlflow.log_metrics({"retrain_roc_auc": roc_auc, "retrain_f1": f1})

        # Fixed: use mlflow.xgboost (not mlflow.sklearn) for XGBClassifier
        mlflow.xgboost.log_model(clf, name="retrained_model")

    result = {
        "status":     "success",
        "roc_auc":    roc_auc,
        "f1":         f1,
        "model_path": str(model_path),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "samples":    len(X_combined),
    }
    return result


# -----------------------------------------------------------
# Main entry point
# -----------------------------------------------------------
def run_drift_and_retrain() -> dict:
    """
    Full pipeline: detect drift, retrain if needed.
    Called by the POST /retrain endpoint and can also be run directly:
        python monitoring/drift_monitor.py
    """
    summary = detect_drift()
    log.info(
        "Drift summary: drifted=%s  share=%.4f  triggered=%s",
        summary["dataset_drifted"],
        summary["drift_share"],
        summary["retrain_triggered"]
    )

    retrain_result = {}
    if summary["retrain_triggered"]:
        log.info("Drift threshold exceeded — starting automated retraining...")
        retrain_result = retrain_model(summary)
    else:
        log.info("No significant drift detected — skipping retraining.")

    return {"drift": summary, "retrain": retrain_result}


if __name__ == "__main__":
    results = run_drift_and_retrain()
    print(json.dumps(results, indent=2))