"""Isolation Forest Model Training and Evaluation Pipeline.

This module implements the classical machine learning baseline for time-series anomaly detection:
1. Feature Engineering: Maps 3D temporal windows (N x T x D) into an enriched 2D statistical
   feature space (N x 4D) via [mean, std, min, max] pooling across the time window.
2. Robust Preprocessing: Median imputation handles missing sensors, followed by RobustScaler
   (IQR scaling) to resist extreme outlier distortion.
3. Anomaly Decision Boundary: Calibrates the score threshold on held-out validation telemetry
   using the precision-recall curve to maximize F1-score.
4. Comprehensive Diagnostics: Reports Precision, Recall, F1, PR-AUC, and False Alarm Rate (FAR).
5. Root Cause Attribution: Analyzes per-metric deviations from training baselines and aggregates
   them by microservice prefix.
"""

from pathlib import Path
import json
from typing import Any, Optional
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import RobustScaler


def _features(x: np.ndarray) -> np.ndarray:
    """Extracts summary statistics across the temporal window dimension (axis=1).

    Computes [mean, standard deviation, minimum, maximum] for each feature,
    transforming shape (N, T, D) into (N, 4*D).

    Args:
        x: 3D NumPy array of shape (N, T, D).

    Returns:
        np.ndarray: 2D feature matrix of shape (N, 4*D).
    """
    return np.concatenate(
        [
            np.nanmean(x, axis=1),
            np.nanstd(x, axis=1),
            np.nanmin(x, axis=1),
            np.nanmax(x, axis=1),
        ],
        axis=1,
    )


def evaluate_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> dict[str, Any]:
    """Computes comprehensive classification metrics against proxy ground truth labels.

    Args:
        y_true: Binary ground truth labels (0 = normal, 1 = anomalous).
        scores: Continuous anomaly scores produced by the model.
        threshold: Decision boundary for binary classification.

    Returns:
        dict[str, Any]: Precision, Recall, F1, PR-AUC, False Alarm Rate, and total count.
    """
    pred = (scores >= threshold).astype(int)
    p = float(precision_score(y_true, pred, zero_division=0))
    r = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    pr_auc = (
        float(average_precision_score(y_true, scores))
        if len(np.unique(y_true)) > 1
        else None
    )
    normal_mask = y_true == 0
    far = float(pred[normal_mask].mean()) if normal_mask.sum() > 0 else 0.0

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "false_alarm_rate": round(far, 4),
        "total_windows": int(len(y_true)),
    }


def train_isolation_forest(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Fits the Isolation Forest baseline, calibrates threshold, and persists artifacts.

    Args:
        data_dir: Directory containing `train.npz`, `val.npz`, `test.npz`, and `manifest.json`.
        output_dir: Output directory to save `isolation_forest.joblib` and `evaluation.json`.

    Returns:
        dict[str, Any]: Diagnostic report containing threshold, validation metrics, and test results.

    Raises:
        ValueError: If training or validation partitions are empty.
    """
    manifest_file = data_dir / "manifest.json"
    features: list[str] = []
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        features = manifest.get("features", [])

    with np.load(data_dir / "train.npz", allow_pickle=False) as d:
        train_x, train_y = d["X"], d["y"]
    with np.load(data_dir / "val.npz", allow_pickle=False) as d:
        val_x, val_y = d["X"], d["y"]

    if not len(train_x) or not len(val_x):
        raise ValueError("Training and validation partitions must contain valid windows.")

    train_f, val_f = _features(train_x), _features(val_x)
    imputer, scaler = SimpleImputer(strategy="median"), RobustScaler()
    train_f = scaler.fit_transform(imputer.fit_transform(train_f))
    val_f = scaler.transform(imputer.transform(val_f))

    # Baseline feature statistics for Root Cause Attribution deviations
    train_metric_means = np.nanmean(train_x, axis=(0, 1))
    train_metric_stds = np.nanstd(train_x, axis=(0, 1))
    train_metric_stds[train_metric_stds < 1e-6] = 1.0

    model = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(train_f)
    scores = -model.score_samples(val_f)

    # Calibrate decision threshold maximizing F1 on validation split
    if len(np.unique(val_y)) == 2:
        p, r, thresholds = precision_recall_curve(val_y, scores)
        f1_vals = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
        threshold = float(thresholds[int(np.argmax(f1_vals))])
    else:
        threshold = float(np.quantile(-model.score_samples(train_f), 0.95))

    val_metrics = evaluate_predictions(val_y, scores, threshold)
    pred = (scores >= threshold).astype(int)

    report: dict[str, Any] = {
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "classification_report": classification_report(val_y, pred, output_dict=True, zero_division=0),
        "caveat": "Injection-time proxy labels; interpret metrics cautiously.",
    }

    # Evaluate on held-out test split
    test_file = data_dir / "test.npz"
    if test_file.exists():
        with np.load(test_file, allow_pickle=False) as d:
            test_x, test_y = d["X"], d["y"]
        if len(test_x):
            test_f = scaler.transform(imputer.transform(_features(test_x)))
            test_scores = -model.score_samples(test_f)
            report["test_metrics"] = evaluate_predictions(test_y, test_scores, threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_data = {
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "threshold": threshold,
        "features": features,
        "train_metric_means": train_metric_means,
        "train_metric_stds": train_metric_stds,
    }
    joblib.dump(artifact_data, output_dir / "isolation_forest.joblib")
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def attribute_root_cause(
    features: list[str],
    deviations: np.ndarray,
    top_k: int = 3
) -> dict[str, Any]:
    """Aggregates per-metric deviations by microservice name and ranks top root causes.

    Args:
        features: Feature names corresponding to the deviation indices.
        deviations: 1D array of metric deviations from baseline.
        top_k: Maximum number of suspect microservices to return.

    Returns:
        dict[str, Any]: Ranked suspect microservices with confidence percentages.
    """
    service_scores: dict[str, float] = {}
    metric_attributions: list[dict[str, Any]] = []

    for col, dev in zip(features, deviations):
        service = col.split("_")[0]
        service_scores[service] = service_scores.get(service, 0.0) + float(dev)
        metric_attributions.append({
            "metric": col,
            "service": service,
            "deviation": round(float(dev), 3),
        })

    total = sum(service_scores.values()) or 1.0
    ranked_services = sorted(
        [
            {
                "service": s,
                "attribution_score": round(sc, 3),
                "attribution_share_pct": round((sc / total) * 100, 1),
                "confidence_pct": round((sc / total) * 100, 1),  # Backward compatibility alias
            }
            for s, sc in service_scores.items()
        ],
        key=lambda x: x["attribution_score"],
        reverse=True,
    )
    metric_attributions.sort(key=lambda x: x["deviation"], reverse=True)

    return {
        "suspect_services": ranked_services[:top_k],
        "top_contributing_metrics": metric_attributions[:5],
        "attribution_method": "metric_deviation_decomposition",
        "attribution_note": "attribution_share_pct represents the relative proportion of observed deviation, not a Bayesian posterior probability."
    }
