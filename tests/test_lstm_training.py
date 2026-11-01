"""Unit tests for ml_engine/lstm_autoencoder.py."""

import json
from pathlib import Path

import numpy as np
import torch

from ml_engine.lstm_autoencoder import LSTMAutoencoder, train_autoencoder


def test_lstm_autoencoder_forward():
    """Verifies forward pass through LSTM encoder and decoder."""
    model = LSTMAutoencoder(n_features=4, hidden_size=8, latent_size=4)
    # Batch size: 2, seq len: 30, features: 4
    x = torch.randn(2, 30, 4)
    out = model(x)
    assert out.shape == x.shape


def test_train_lstm_autoencoder_flow(tmp_path: Path):
    """Verifies lightweight end-to-end training of the LSTM Autoencoder."""
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir(parents=True)
    output_file = output_dir / "lstm_autoencoder.pt"

    n_features = 3
    train_x = np.random.randn(10, 30, n_features).astype(np.float32)
    train_y = np.zeros(10, dtype=int)
    val_x = np.random.randn(5, 30, n_features).astype(np.float32)
    val_y = np.array([0, 0, 0, 1, 1], dtype=int)
    test_x = np.random.randn(5, 30, n_features).astype(np.float32)
    test_y = np.array([0, 0, 0, 1, 1], dtype=int)

    features = ["svc_cpu", "svc_mem", "svc_net"]
    (data_dir / "manifest.json").write_text(json.dumps({"features": features}), encoding="utf-8")
    np.savez(data_dir / "train.npz", X=train_x, y=train_y)
    np.savez(data_dir / "val.npz", X=val_x, y=val_y)
    np.savez(data_dir / "test.npz", X=test_x, y=test_y)

    report = train_autoencoder(
        data_dir=data_dir,
        output_file=output_file,
        epochs=1,
        batch_size=4
    )

    assert "threshold" in report
    assert "validation_metrics" in report
    assert output_file.exists()
    assert (output_dir / "lstm_evaluation.json").exists()
