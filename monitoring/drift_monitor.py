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
    reference_path: str | Path = DATA_DIR / "X_train.csv",
    current_path:   str | Path = DATA_DIR / "X_test.csv",
    threshold: float = DRIFT_THRESHOLD,
) -> dict:
    """
    Run Evidently drift detection between reference and current datasets.
    Returns a summary dict and writes JSON + HTML reports.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
    except ImportError:
        log.error("evidently is not installed. Run: pip install evidently")
        raise

    log.info("Loading reference data from %s", reference_path)
    reference = pd.read_csv(reference_path)
    current   = pd.read_csv(current_path)

    log.info("Running Evidently DataDriftPreset (%d ref rows, %d cur rows)",
             len(reference), len(current))

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    result = report.as_dict()

    # Parse per-feature drift results
    feature_drift: dict[str, bool] = {}
    try:
        metrics = result["metrics"]
        for metric in metrics:
            if metric.get("metric") == "DataDriftTable":
                drift_by_columns = metric["result"].get("drift_by_columns", {})
                for col, info in drift_by_columns.items():
                    feature_drift[col] = bool(info.get("drift_detected", False))
                break
    except (KeyError, TypeError) as e:
        log.warning("Could not parse per-feature drift: %s", e)

    drifted     = sum(1 for v in feature_drift.values() if v)
    total       = len(feature_drift) or 1
    drift_share = round(drifted / total, 4)
    drifted_flag = drift_share > threshold

    summary = {
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "dataset_drifted":   drifted_flag,
        "drifted_features":  drifted,
        "total_features":    total,
        "drift_share":       drift_share,
        "threshold":         threshold,
        "retrain_triggered": drifted_flag,
        "feature_drift":     feature_drift,
    }

    # Save JSON summary
    with open(DRIFT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Drift summary saved to %s", DRIFT_SUMMARY)

    # Save HTML report
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    html_path = REPORTS_DIR / f"drift_report_{ts}.html"
    report.save_html(str(html_path))
    log.info("Drift HTML report saved to %s", html_path)

    return summary


# -----------------------------------------------------------
# Automated retraining
# -----------------------------------------------------------
def retrain_model(summary: dict | None = None) -> dict:
    """
    Retrain the XGBoost model on the combined train+test data.
    Logs the run to MLflow and saves the new model to models/best_model.pkl.
    Returns a result dict with new metrics.
    """
    from sklearn.metrics import roc_auc_score, f1_score
    from xgboost import XGBClassifier

    log.info("Loading training data for retraining...")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    X_test  = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    # Combine train + test for retraining (simulate new incoming data)
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True)

    params = {
        "n_estimators":    300,
        "max_depth":       6,
        "learning_rate":   0.05,
        "scale_pos_weight": int((y_combined == 0).sum() / (y_combined == 1).sum()),
        "subsample":       0.8,
        "colsample_bytree": 0.8,
        "random_state":    42,
        "n_jobs":          -1,
        "eval_metric":     "logloss",
    }

    log.info("Retraining XGBoost on %d samples...", len(X_combined))
    clf = XGBClassifier(**params)
    clf.fit(X_combined, y_combined, verbose=False)

    # Evaluate on held-out test set
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    roc_auc = round(float(roc_auc_score(y_test, y_proba)), 4)
    f1      = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)

    log.info("Retrained model — ROC-AUC: %.4f  F1: %.4f", roc_auc, f1)

    # Save model
    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(clf, model_path)
    log.info("Updated model saved to %s", model_path)

    # Log to MLflow
    with mlflow.start_run(run_name="auto_retrain"):
        mlflow.log_params(params)
        mlflow.log_params({
            "trigger":       "drift_monitor",
            "drift_share":   summary.get("drift_share", "N/A") if summary else "manual",
            "drifted_feats": summary.get("drifted_features", "N/A") if summary else "manual",
        })
        mlflow.log_metrics({"retrain_roc_auc": roc_auc, "retrain_f1": f1})
        mlflow.sklearn.log_model(clf, "model")

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
# Main entry point for running directly
# -----------------------------------------------------------
def run_drift_and_retrain() -> dict:
    """
    Full pipeline: detect drift, retrain if needed.
    Call this from a scheduler (APScheduler / cron) or the /retrain endpoint.
    """
    summary = detect_drift()
    log.info("Drift summary: drifted=%s share=%.4f triggered=%s",
             summary["dataset_drifted"], summary["drift_share"], summary["retrain_triggered"])

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
