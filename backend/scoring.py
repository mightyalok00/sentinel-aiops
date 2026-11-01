"""Telemetry Anomaly Scoring and Root Cause Analysis (RCA) Engine.

This module provides production-grade inference pipelines for telemetry anomaly scoring:
1. Model Artifact Loading: Singleton in-memory caching of pre-trained Isolation Forest
   and PyTorch LSTM Autoencoder artifacts.
2. Dynamic Feature Alignment: Ensures incoming streaming batches or windows are strictly
   aligned with the schema established during training.
3. Root Cause Attribution (RCA): Decomposes window-level and temporal reconstruction errors
   across individual metrics and aggregates them by microservice topology to quantify
   culprit service confidence.
"""

from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd
import torch

# Base project directory resolution
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"

# Singleton model cache to avoid repeated disk I/O on concurrent requests
_MODEL_CACHE: dict[str, Any] = {}


def load_isolation_forest() -> Optional[dict[str, Any]]:
    """Loads and caches the pre-trained Isolation Forest artifact from disk.

    Returns:
        Optional[dict[str, Any]]: Dictionary containing the imputer, scaler,
        Isolation Forest model, baseline statistics, and calibrated threshold,
        or None if the artifact does not exist.
    """
    if "iforest" not in _MODEL_CACHE:
        model_path = ARTIFACTS_DIR / "isolation_forest.joblib"
        if model_path.exists():
            _MODEL_CACHE["iforest"] = joblib.load(model_path)
    return _MODEL_CACHE.get("iforest")


def load_lstm_autoencoder() -> Optional[dict[str, Any]]:
    """Loads and caches the pre-trained PyTorch LSTM Autoencoder model.

    Initializes the network architecture, transfers trained weights to evaluation mode,
    and returns normalization vectors and calibrated anomaly thresholds.

    Returns:
        Optional[dict[str, Any]]: Dictionary containing the instantiated model,
        feature list, mean/std vectors, baseline medians, and threshold, or None
        if artifact is missing.
    """
    if "lstm" not in _MODEL_CACHE:
        model_path = ARTIFACTS_DIR / "lstm_autoencoder.pt"
        if model_path.exists():
            from ml_engine.lstm_autoencoder import LSTMAutoencoder
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            model = LSTMAutoencoder(checkpoint["n_features"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            _MODEL_CACHE["lstm"] = {
                "model": model,
                "features": checkpoint.get("features", []),
                "mean": np.array(checkpoint["mean"], dtype=np.float32),
                "std": np.array(checkpoint["std"], dtype=np.float32),
                "medians": np.array(checkpoint["train_medians"], dtype=np.float32),
                "threshold": float(checkpoint["threshold"])
            }
    return _MODEL_CACHE.get("lstm")


def _aggregate_root_cause(
    features: list[str],
    metric_scores: np.ndarray,
    top_k: int = 3
) -> dict[str, Any]:
    """Aggregates metric-level anomaly deviations by microservice to rank root causes.

    Each metric name is assumed to follow the topology convention `<service>_<metric_name>`
    (e.g., `checkoutservice_cpu`, `cartservice_latency`). Deviations are summed per service
    and normalized into percentage confidence scores.

    Args:
        features: List of feature names corresponding to indices in `metric_scores`.
        metric_scores: 1D NumPy array representing individual feature deviations or reconstruction errors.
        top_k: Maximum number of suspect microservices to return in ranked order.

    Returns:
        dict[str, Any]: Ranked suspect services with attribution scores and confidence percentages,
        along with the top contributing individual metrics.
    """
    service_scores: dict[str, float] = {}
    metric_attributions: list[dict[str, Any]] = []

    for col, score in zip(features, metric_scores):
        service = col.split("_")[0]
        service_scores[service] = service_scores.get(service, 0.0) + float(score)
        metric_attributions.append({
            "metric": col,
            "service": service,
            "anomaly_contribution": round(float(score), 4)
        })

    total = sum(service_scores.values()) or 1.0
    ranked_services = sorted(
        [
            {
                "service": s,
                "score": round(sc, 4),
                "confidence_pct": round((sc / total) * 100, 1)
            }
            for s, sc in service_scores.items()
        ],
        key=lambda x: x["score"],
        reverse=True
    )
    metric_attributions.sort(key=lambda x: x["anomaly_contribution"], reverse=True)

    return {
        "suspect_services": ranked_services[:top_k],
        "top_contributing_metrics": metric_attributions[:5]
    }


def score_rows(
    rows: list[dict[str, Any]],
    model_type: str = "lstm",
    custom_threshold: Optional[float] = None
) -> dict[str, Any]:
    """Infers anomaly scores and executes Root Cause Analysis on incoming telemetry batches.

    Execution Pipeline:
    1. Validation: Ensures input rows contain valid numeric metric values.
    2. Alignment: Maps row columns to the feature schema expected by the active model.
    3. Inference:
       - LSTM Autoencoder: Computes temporal reconstruction loss MSE(x, x_hat).
       - Isolation Forest: Computes window summary statistics [mean, std, min, max] and scores samples.
    4. Attribution: Aggregates errors by microservice to output ranked culprits.

    Args:
        rows: Sequence of metric records (dictionary per timestamp/step).
        model_type: Selected model ('lstm' or 'iforest').
        custom_threshold: Optional override for the calibrated anomaly decision boundary.

    Returns:
        dict[str, Any]: Response dictionary with anomaly scores, calibrated threshold,
        feature counts, model identification, and root-cause breakdown.

    Raises:
        ValueError: If input is empty, lacks numeric fields, or consists solely of NaNs/Infs.
    """
    if not rows:
        raise ValueError("At least one row of metrics is required for evaluation.")

    frame = pd.DataFrame(rows).select_dtypes(include="number")
    frame = frame.drop(columns=["time"], errors="ignore")

    if frame.empty:
        raise ValueError("At least one numeric metric column is required.")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if frame.isna().all().all():
        raise ValueError("All numeric metric values are missing or infinite.")

    # -------------------------------------------------------------------------
    # Route 1: PyTorch LSTM Autoencoder (Temporal Sequence Reconstruction)
    # -------------------------------------------------------------------------
    if model_type.lower() == "lstm":
        lstm_data = load_lstm_autoencoder()
        if lstm_data:
            features = lstm_data["features"]
            if features:
                aligned = frame.reindex(columns=features, fill_value=np.nan).to_numpy(dtype=np.float32)
            else:
                aligned = frame.to_numpy(dtype=np.float32)
                features = list(frame.columns)

            medians = lstm_data["medians"]
            mean = lstm_data["mean"]
            std = lstm_data["std"]
            threshold = (
                custom_threshold
                if custom_threshold is not None and custom_threshold > 0
                else lstm_data["threshold"]
            )

            # Impute missing values with baseline medians and standardize
            X_filled = np.where(np.isfinite(aligned), aligned, medians[None, :])
            X_norm = ((X_filled - mean) / std).astype(np.float32)

            # Model expects batch shape: (batch_size=1, sequence_length, n_features)
            tensor_in = torch.from_numpy(X_norm).unsqueeze(0)
            with torch.no_grad():
                recon = lstm_data["model"](tensor_in).squeeze(0).numpy()

            # Point-wise squared reconstruction error e_it = (x_it - x_hat_it)^2
            diff = (X_norm - recon) ** 2
            row_scores = [float(s) for s in diff.mean(axis=1)]
            metric_errors = diff.mean(axis=0)
            rca = _aggregate_root_cause(features, metric_errors)

            return {
                "model": "LSTM Autoencoder (Pre-trained)",
                "scores": row_scores,
                "overall_score": float(np.mean(row_scores)),
                "threshold": threshold,
                "feature_count": len(features),
                "root_cause_analysis": rca
            }

    # -------------------------------------------------------------------------
    # Route 2: Isolation Forest (Statistical Summary Feature Space)
    # -------------------------------------------------------------------------
    iforest_data = load_isolation_forest()
    if iforest_data:
        features = iforest_data.get("features", [])
        if features:
            aligned = frame.reindex(columns=features, fill_value=np.nan).to_numpy(dtype=np.float32)
        else:
            aligned = frame.to_numpy(dtype=np.float32)
            features = list(frame.columns)

        threshold = (
            custom_threshold
            if custom_threshold is not None and custom_threshold > 0
            else iforest_data["threshold"]
        )
        imputer = iforest_data["imputer"]
        scaler = iforest_data["scaler"]
        model = iforest_data["model"]

        # Compute summary statistics across the sliding window
        if len(aligned) == 1:
            row_stats = np.concatenate([aligned, np.zeros_like(aligned), aligned, aligned], axis=1)
        else:
            w_mean = np.nanmean(aligned, axis=0, keepdims=True)
            w_std = np.nanstd(aligned, axis=0, keepdims=True)
            w_min = np.nanmin(aligned, axis=0, keepdims=True)
            w_max = np.nanmax(aligned, axis=0, keepdims=True)
            row_stats = np.concatenate([w_mean, w_std, w_min, w_max], axis=1)

        imputed_scaled = scaler.transform(imputer.transform(row_stats))
        score_val = float(-model.score_samples(imputed_scaled)[0])

        # Feature z-score deviations from normal training baseline
        means = iforest_data.get("train_metric_means", np.zeros(len(features)))
        stds = iforest_data.get("train_metric_stds", np.ones(len(features)))
        row_mean = np.nanmean(aligned, axis=0)
        deviations = np.abs(np.nan_to_num(row_mean - means, nan=0.0)) / (stds + 1e-6)
        rca = _aggregate_root_cause(features, deviations)

        scores = [score_val] * len(frame)
        return {
            "model": "Isolation Forest (Pre-trained)",
            "scores": scores,
            "overall_score": score_val,
            "threshold": threshold,
            "feature_count": len(features),
            "root_cause_analysis": rca
        }

    # -------------------------------------------------------------------------
    # Route 3: Graceful Fallback (Local Uncalibrated Baseline)
    # -------------------------------------------------------------------------
    from sklearn.ensemble import IsolationForest as FallbackIF
    from sklearn.impute import SimpleImputer as FallbackImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler as FallbackScaler

    values = frame.to_numpy(dtype=float)
    pipeline = make_pipeline(
        FallbackImputer(strategy="median"),
        FallbackScaler(),
        FallbackIF(n_estimators=100, random_state=42, contamination="auto")
    )
    pipeline.fit(values)
    model = pipeline[-1]
    scores = -model.score_samples(pipeline[:-1].transform(values))
    threshold = float(np.quantile(scores, 0.95))
    features = list(frame.columns)
    deviations = np.nanstd(values, axis=0)
    rca = _aggregate_root_cause(features, deviations)

    return {
        "model": "Ad-hoc Fallback Isolation Forest (Uncalibrated)",
        "scores": [float(x) for x in scores],
        "overall_score": float(np.mean(scores)),
        "threshold": threshold,
        "feature_count": int(frame.shape[1]),
        "root_cause_analysis": rca
    }
