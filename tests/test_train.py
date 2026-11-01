"""Unit and integration tests for ml_engine/train.py."""

import json
from pathlib import Path

import numpy as np

from ml_engine.train import _features, attribute_root_cause, evaluate_predictions, train_isolation_forest


def test_features_extraction():
    """Verifies that sliding window feature extraction computes correct summary statistics."""
    # Shape: (5 windows, 30 timesteps, 2 features)
    X = np.ones((5, 30, 2), dtype=np.float32)
    feats = _features(X)
    # mean, std, min, max -> 2 * 4 = 8 features
    assert feats.shape == (5, 8)
    assert np.all(feats[:, 0] == 1.0)  # mean
    assert np.all(feats[:, 2] == 0.0)  # std


def test_evaluate_predictions():
    """Verifies precision, recall, F1, PR-AUC, and false alarm rate calculations."""
    y_true = np.array([0, 0, 0, 1, 1], dtype=int)
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    metrics = evaluate_predictions(y_true, scores, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["false_alarm_rate"] == 0.0
    assert metrics["total_windows"] == 5
    assert metrics["pr_auc"] is not None


def test_attribute_root_cause():
    """Verifies root cause deviation aggregation and ranking."""
    features = ["adservice_cpu", "adservice_mem", "cartservice_cpu"]
    deviations = np.array([5.0, 3.0, 1.0])

    rca = attribute_root_cause(features, deviations, top_k=2)
    suspects = rca["suspect_services"]
    assert len(suspects) == 2
    assert suspects[0]["service"] == "adservice"
    assert suspects[0]["attribution_score"] == 8.0
    assert suspects[1]["service"] == "cartservice"
    assert suspects[1]["attribution_score"] == 1.0
    assert "attribution_share_pct" in suspects[0]


def test_train_isolation_forest_flow(tmp_path: Path):
    """Verifies end-to-end training of the Isolation Forest baseline on synthetic data."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir(parents=True)

    # Create dummy windows
    n_samples, win_len, n_feats = 20, 30, 4
    train_x = np.random.randn(n_samples, win_len, n_feats).astype(np.float32)
    val_x = np.random.randn(10, win_len, n_feats).astype(np.float32)
    val_y = np.array([0] * 8 + [1] * 2, dtype=int)
    test_x = np.random.randn(10, win_len, n_feats).astype(np.float32)
    test_y = np.array([0] * 8 + [1] * 2, dtype=int)

    features = ["svc1_cpu", "svc1_mem", "svc2_cpu", "svc2_mem"]
    (data_dir / "manifest.json").write_text(json.dumps({"features": features}), encoding="utf-8")
    np.savez(data_dir / "train.npz", X=train_x)
    np.savez(data_dir / "val.npz", X=val_x, y=val_y)
    np.savez(data_dir / "test.npz", X=test_x, y=test_y)

    report = train_isolation_forest(data_dir, output_dir)
    assert "threshold" in report
    assert "validation_metrics" in report
    assert "test_metrics" in report
    assert (output_dir / "isolation_forest.joblib").exists()
    assert (output_dir / "evaluation.json").exists()
