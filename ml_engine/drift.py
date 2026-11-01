"""Statistical Telemetry Drift Monitoring & Distribution Shift Detection.

Monitors incoming live telemetry streams for distribution drift against the established
training baseline using:
1. Population Stability Index (PSI): Standard industry measure of distribution shift.
   - PSI < 0.10: Stable (No shift).
   - 0.10 <= PSI < 0.25: Moderate Drift (Distribution shift warning).
   - PSI >= 0.25: Critical Drift (Significant change; model recalibration recommended).
2. 2-Sample Kolmogorov-Smirnov (KS) Test: Non-parametric test detecting deviations in
   empirical cumulative distribution functions (ECDF) with p-value thresholding (p < 0.01).
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from backend.scoring import load_isolation_forest, load_lstm_autoencoder


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """Computes the Population Stability Index (PSI) between expected baseline and actual samples.

    Args:
        expected: 1D array of reference baseline feature values.
        actual: 1D array of incoming / current feature values.
        num_bins: Number of quantile bins for discretization.
        epsilon: Small smoothing constant to prevent division by zero or log(0).

    Returns:
        float: Population Stability Index value.
    """
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) < 5 or len(actual) < 5:
        return 0.0

    # Determine quantile bins from baseline
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Ensure strictly increasing bins
    for i in range(1, len(bins) - 1):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-5

    # Compute frequency distributions
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = (expected_counts + epsilon) / (np.sum(expected_counts) + epsilon * num_bins)
    actual_pct = (actual_counts + epsilon) / (np.sum(actual_counts) + epsilon * num_bins)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.nan_to_num(psi_val, nan=0.0))


def monitor_telemetry_drift(
    rows: list[dict[str, Any]],
    alpha: float = 0.01
) -> dict[str, Any]:
    """Analyzes a batch of incoming telemetry rows for statistical distribution drift.

    Args:
        rows: Incoming list of telemetry dictionaries.
        alpha: Significance level for Kolmogorov-Smirnov test (default: 0.01).

    Returns:
        dict[str, Any]: Comprehensive drift report including overall PSI, drifted features,
        per-feature metrics, and operational recommendations.
    """
    if not rows:
        return {
            "drift_detected": False,
            "drift_severity": "none",
            "overall_psi": 0.0,
            "drifted_features": [],
            "message": "Empty telemetry payload; no drift evaluation performed."
        }

    # Retrieve baseline statistics from cached model artifacts
    lstm_data = load_lstm_autoencoder()
    iforest_data = load_isolation_forest()

    features: list[str] = []
    baseline_means: np.ndarray = np.array([])
    baseline_stds: np.ndarray = np.array([])

    if lstm_data and "features" in lstm_data:
        features = lstm_data["features"]
        baseline_means = lstm_data["mean"]
        baseline_stds = lstm_data["std"]
    elif iforest_data and "features" in iforest_data:
        features = iforest_data["features"]
        baseline_means = iforest_data.get("train_metric_means", np.zeros(len(features)))
        baseline_stds = iforest_data.get("train_metric_stds", np.ones(len(features)))

    if len(features) == 0:
        return {
            "drift_detected": False,
            "drift_severity": "none",
            "overall_psi": 0.0,
            "drifted_features": [],
            "message": "No trained model baseline available for drift comparison."
        }

    df = pd.DataFrame(rows)
    feature_psi_scores: list[float] = []
    drifted_features: list[dict[str, Any]] = []

    for idx, feat in enumerate(features):
        if feat not in df.columns:
            continue

        actual_vals = pd.to_numeric(df[feat], errors="coerce").dropna().to_numpy()
        if len(actual_vals) < 5:
            continue

        b_mean = float(baseline_means[idx]) if idx < len(baseline_means) else 0.0
        b_std = float(baseline_stds[idx]) if idx < len(baseline_stds) else 1.0
        if b_std <= 1e-6:
            b_std = 1.0

        # Synthetic baseline distribution from stored moments
        np.random.seed(42)
        simulated_baseline = np.random.normal(loc=b_mean, scale=b_std, size=max(len(actual_vals) * 2, 100))

        # Compute PSI
        psi = compute_psi(simulated_baseline, actual_vals)
        feature_psi_scores.append(psi)

        # 2-sample KS test
        ks_stat, p_value = stats.ks_2samp(actual_vals, simulated_baseline)

        if psi >= 0.25 or (p_value < alpha and ks_stat > 0.40):
            service = feat.split("_")[0]
            drifted_features.append({
                "feature": feat,
                "service": service,
                "psi": round(psi, 3),
                "ks_statistic": round(float(ks_stat), 3),
                "p_value": float(f"{p_value:.2e}"),
                "actual_mean": round(float(np.mean(actual_vals)), 3),
                "baseline_mean": round(b_mean, 3)
            })

    avg_psi = float(np.mean(feature_psi_scores)) if feature_psi_scores else 0.0

    if avg_psi >= 0.25 or len(drifted_features) >= 5:
        severity = "critical"
        drift_detected = True
        recommendation = "Critical telemetry drift detected. Recalibrate anomaly thresholds or retrain models on recent data."
    elif avg_psi >= 0.10 or len(drifted_features) > 0:
        severity = "moderate"
        drift_detected = True
        recommendation = "Moderate telemetry drift observed. Monitor false positive rates and consider scheduled retraining."
    else:
        severity = "none"
        drift_detected = False
        recommendation = "Telemetry distributions are stable and align with the training baseline."

    return {
        "drift_detected": drift_detected,
        "drift_severity": severity,
        "overall_psi": round(avg_psi, 4),
        "total_features_evaluated": len(feature_psi_scores),
        "drifted_feature_count": len(drifted_features),
        "drifted_features": drifted_features[:10],
        "recommendation": recommendation
    }
