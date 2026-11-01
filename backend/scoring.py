# pyrefly: ignore-errors[missing-source-for-stubs]
"""Telemetry Anomaly Scoring and Root Cause Analysis (RCA) Engine.

This module provides production-grade inference pipelines for telemetry anomaly scoring:
1. Model Artifact Loading: Singleton in-memory caching of pre-trained Isolation Forest
   and PyTorch LSTM Autoencoder artifacts.
2. Dynamic Feature Alignment: Maps incoming raw telemetry records to the established training schema.
3. Training-Consistent Sliding Windowing:
   - Evaluates sequences strictly using 30-timestep sliding windows matching the training pipeline.
   - Padds with baseline medians if input sequence length < 30.
   - Prevents inference sequence length mismatch and score distribution drift.
4. Relative Root Cause Attribution:
   - Decomposes reconstruction errors across individual metrics and aggregates by microservice.
   - Outputs `attribution_share_pct` (relative contribution to error), avoiding misrepresenting
     relative deviation shares as Bayesian posterior probabilities.
"""

import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd  # type: ignore # pyrefly: ignore
import torch
from numpy.lib.stride_tricks import sliding_window_view

# Base project directory resolution
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"

# Singleton model cache to avoid repeated disk I/O on concurrent requests
_MODEL_CACHE: dict[str, Any] = {}

WINDOW_SIZE: int = 30


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
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)  # nosec B614
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
    metric_scores: Any,
    top_k: int = 3
) -> dict[str, Any]:
    """Aggregates metric-level anomaly deviations by microservice to rank root causes.

    Metrics follow the microservice topology convention `<service>_<metric_name>`
    (e.g., `checkoutservice_cpu`, `cartservice_latency`). Deviations are aggregated per service
    and normalized into `attribution_share_pct` (relative share of total deviation).

    Note: `attribution_share_pct` represents relative attribution contribution to the
    observed anomaly deviation, NOT a Bayesian posterior probability.

    Args:
        features: List of feature names corresponding to indices in `metric_scores`.
        metric_scores: 1D NumPy array representing individual feature deviations or reconstruction errors.
        top_k: Maximum number of suspect microservices to return in ranked order.

    Returns:
        dict[str, Any]: Ranked suspect services with attribution scores and attribution share percentages,
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
                "attribution_score": round(sc, 4),
                "attribution_share_pct": round((sc / total) * 100, 1),
                "confidence_pct": round((sc / total) * 100, 1)  # Backward compatibility alias
            }
            for s, sc in service_scores.items()
        ],
        key=lambda x: x["attribution_score"],
        reverse=True
    )
    metric_attributions.sort(key=lambda x: x["anomaly_contribution"], reverse=True)

    # Temperature-scaled Softmax calibration over service scores for calibrated attribution probability
    if ranked_services:
        raw_scores = np.array([item["attribution_score"] for item in ranked_services], dtype=np.float64)
        if len(raw_scores) > 1 and np.std(raw_scores) > 1e-6:
            # Standardize and scale with temperature T=1.5 to prevent overconfidence
            scaled = (raw_scores - np.mean(raw_scores)) / (np.std(raw_scores) * 1.5)
            exp_s = np.exp(scaled - np.max(scaled))
            probs = (exp_s / np.sum(exp_s)) * 100.0
        else:
            probs = np.full(len(ranked_services), 100.0 / max(len(ranked_services), 1))

        for idx, item in enumerate(ranked_services):
            item["calibrated_probability_pct"] = round(float(probs[idx]), 1)

    return {
        "suspect_services": ranked_services[:top_k],
        "top_contributing_metrics": metric_attributions[:5],
        "attribution_method": "reconstruction_error_decomposition",
        "attribution_note": "attribution_share_pct indicates relative error decomposition; calibrated_probability_pct provides temperature-scaled Softmax confidence."
    }


def score_rows(
    rows: list[dict[str, Any]],
    model_type: str = "lstm",
    custom_threshold: Optional[float] = None
) -> dict[str, Any]:
    """Infers anomaly scores and executes Root Cause Analysis on incoming telemetry batches.

    Enforces 30-timestep sliding windows to match the training pipeline distribution:
    - If len(rows) < 30: Pads leading timestamps with baseline medians so sequence length is 30.
    - If len(rows) >= 30: Slides 30-step temporal windows across time, scoring each window and
      aligning step scores to the calibrated threshold.

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

            N = len(aligned)

            # Enforce 30-timestep windowing compatible with training:
            if N < WINDOW_SIZE:
                # Pad leading timesteps with baseline medians
                pad_len = WINDOW_SIZE - N
                pad_array = np.tile(medians[None, :], (pad_len, 1)).astype(np.float32)
                full_seq = np.vstack([pad_array, aligned])
                X_filled = np.where(np.isfinite(full_seq), full_seq, medians[None, :])
                X_norm = ((X_filled - mean) / std).astype(np.float32)

                tensor_in = torch.as_tensor(X_norm).unsqueeze(0)
                with torch.no_grad():
                    recon = lstm_data["model"](tensor_in).squeeze(0).numpy()

                diff = (X_norm - recon) ** 2
                diff_actual = diff[pad_len:]
                row_scores = [float(s) for s in diff_actual.mean(axis=1)]
                metric_errors = diff_actual.mean(axis=0)
            else:
                # Sliding 30-timestep windows across sequence
                X_filled = np.where(np.isfinite(aligned), aligned, medians[None, :])
                X_norm = ((X_filled - mean) / std).astype(np.float32)

                windows = sliding_window_view(X_norm, (WINDOW_SIZE, X_norm.shape[1])).squeeze(1).copy()

                tensor_in = torch.as_tensor(windows)
                with torch.no_grad():
                    recon = lstm_data["model"](tensor_in).numpy()

                diff = (windows - recon) ** 2
                window_scores = diff.mean(axis=(1, 2))  # shape: (num_windows,)

                # Map window scores back to timestamps:
                # Timestamps 0..WINDOW_SIZE-2 inherit window 0's score; subsequent timestamps get their window's score
                row_scores = [float(window_scores[0])] * (WINDOW_SIZE - 1) + [float(s) for s in window_scores]
                metric_errors = diff.mean(axis=(0, 1))

            rca = _aggregate_root_cause(features, metric_errors)

            return {
                "model": "LSTM Autoencoder (Pre-trained)",
                "scores": row_scores,
                "overall_score": float(np.mean(row_scores)),
                "threshold": threshold,
                "feature_count": len(features),
                "window_size_used": WINDOW_SIZE,
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

        N = len(aligned)
        if N < WINDOW_SIZE:
            # Impute and pad
            medians = iforest_data.get("train_metric_means", np.zeros(len(features)))
            pad_len = WINDOW_SIZE - N
            pad_array = np.tile(medians[None, :], (pad_len, 1)).astype(np.float32)
            full_seq = np.vstack([pad_array, aligned])
            w_mean = np.nanmean(full_seq, axis=0, keepdims=True)
            w_std = np.nanstd(full_seq, axis=0, keepdims=True)
            w_min = np.nanmin(full_seq, axis=0, keepdims=True)
            w_max = np.nanmax(full_seq, axis=0, keepdims=True)
            row_stats = np.concatenate([w_mean, w_std, w_min, w_max], axis=1)
            imputed_scaled = scaler.transform(imputer.transform(row_stats))
            score_val = float(-model.score_samples(imputed_scaled)[0])
            scores = [score_val] * N
        else:
            # Compute 30-step sliding summary features
            windows = sliding_window_view(aligned, (WINDOW_SIZE, aligned.shape[1])).squeeze(1)
            w_mean = np.nanmean(windows, axis=1)
            w_std = np.nanstd(windows, axis=1)
            w_min = np.nanmin(windows, axis=1)
            w_max = np.nanmax(windows, axis=1)
            row_stats = np.concatenate([w_mean, w_std, w_min, w_max], axis=1)
            imputed_scaled = scaler.transform(imputer.transform(row_stats))
            w_scores = -model.score_samples(imputed_scaled)
            scores = [float(w_scores[0])] * (WINDOW_SIZE - 1) + [float(s) for s in w_scores]

        # Feature z-score deviations from normal training baseline
        means = iforest_data.get("train_metric_means", np.zeros(len(features)))
        stds = iforest_data.get("train_metric_stds", np.ones(len(features)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_mean = np.nanmean(aligned, axis=0)
            row_mean = np.nan_to_num(row_mean, nan=0.0)
        deviations = np.abs(row_mean - means) / (stds + 1e-6)
        rca = _aggregate_root_cause(features, deviations)

        return {
            "model": "Isolation Forest (Pre-trained)",
            "scores": scores,
            "overall_score": float(np.mean(scores)),
            "threshold": threshold,
            "feature_count": len(features),
            "window_size_used": WINDOW_SIZE,
            "root_cause_analysis": rca
        }

    # -------------------------------------------------------------------------
    # Route 3: Graceful Fallback (Local Uncalibrated Baseline)
    # -------------------------------------------------------------------------
    from sklearn.ensemble import IsolationForest as FallbackIF  # type: ignore # pyrefly: ignore
    from sklearn.impute import SimpleImputer as FallbackImputer  # type: ignore # pyrefly: ignore
    from sklearn.pipeline import make_pipeline  # type: ignore # pyrefly: ignore
    from sklearn.preprocessing import RobustScaler as FallbackScaler  # type: ignore # pyrefly: ignore

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
    deviations = frame.std(axis=0, ddof=0).fillna(0.0).to_numpy(dtype=float)
    rca = _aggregate_root_cause(features, deviations)

    return {
        "model": "Ad-hoc Fallback Isolation Forest (Uncalibrated)",
        "scores": [float(x) for x in scores],
        "overall_score": float(np.mean(scores)),
        "threshold": threshold,
        "feature_count": frame.shape[1],
        "root_cause_analysis": rca
    }
