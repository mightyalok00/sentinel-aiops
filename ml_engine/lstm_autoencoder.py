"""PyTorch Long Short-Term Memory (LSTM) Autoencoder for Telemetry Anomaly Detection.

Mathematical Architecture:
1. Encoder: An LSTM network maps an input temporal window X in R^(T x D) (T time steps,
   D features) into a hidden sequence, extracting the final hidden state h_T in R^H.
2. Latent Bottleneck: A linear projection compresses h_T into a latent representation z in R^L (L < H).
3. Decoder: z is projected back to R^H, repeated across T time steps, and fed into a decoder
   LSTM to reconstruct the original temporal window X_hat in R^(T x D).
4. Anomaly Objective: The network trains exclusively on normal operational telemetry using
   Mean Squared Error (MSE):
       L(X, X_hat) = (1 / (T * D)) sum_{t=1}^T sum_{d=1}^D (X_{t, d} - X_hat_{t, d})^2
   At inference, uncharacteristic patterns yield high reconstruction error, signaling anomalies.
"""

import json
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence temporal autoencoder using LSTM recurrent cells."""

    def __init__(self, n_features: int, hidden_size: int = 32, latent_size: int = 16) -> None:
        """Initializes the encoder-decoder LSTM autoencoder.

        Args:
            n_features: Number of telemetry metric features (input dimension D).
            hidden_size: Hidden state dimension of encoder and decoder LSTMs.
            latent_size: Dimension of the compressed information bottleneck layer.
        """
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes forward pass: encodes sequence, compresses, decodes, and reconstructs.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features).

        Returns:
            torch.Tensor: Reconstructed sequence of identical shape (batch_size, sequence_length, n_features).
        """
        # Encode: extract final hidden state
        _, (h, _) = self.encoder(x)
        # Latent projection: h[-1] shape (batch_size, hidden_size) -> (batch_size, latent_size)
        z = self.to_latent(h[-1])
        # Expand latent vector across the sequence length T
        repeated = self.from_latent(z).unsqueeze(1).expand(-1, x.size(1), -1)
        # Decode sequence
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def _normalize_windows(
    X: np.ndarray,
    medians: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray
) -> np.ndarray:
    """Imputes missing values with feature medians and applies z-score standardization.

    Args:
        X: Raw 3D window array of shape (N, T, D).
        medians: 1D baseline median vector of shape (D,).
        mean: Normal baseline mean vector of shape (1, 1, D).
        std: Normal baseline standard deviation vector of shape (1, 1, D).

    Returns:
        np.ndarray: Normalized 3D array of shape (N, T, D) with float32 precision.
    """
    X_filled = np.where(np.isfinite(X), X, medians[None, None, :])
    return ((X_filled - mean) / std).astype(np.float32)


def compute_reconstruction_errors(
    model: nn.Module,
    X_norm: np.ndarray,
    device: torch.device,
    batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes overall sample MSE and per-feature reconstruction errors in batches.

    Args:
        model: Trained LSTMAutoencoder instance.
        X_norm: Normalized 3D numpy array of shape (N, T, D).
        device: PyTorch device ('cuda' or 'cpu').
        batch_size: Mini-batch size for forward inference.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Sample errors: 1D array of shape (N,) with mean squared error per window.
            - Metric errors: 2D array of shape (N, D) with mean squared error per metric across time.
    """
    model.eval()
    errors: list[float] = []
    metric_errors: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(X_norm), batch_size):
            batch = torch.from_numpy(X_norm[i:i + batch_size]).to(device)
            recon = model(batch)
            diff = (batch - recon) ** 2

            # Window anomaly score: mean over sequence length (dim 1) and features (dim 2)
            sample_errors = diff.mean(dim=(1, 2)).cpu().numpy()
            errors.extend(sample_errors)

            # Feature error for RCA: mean over sequence length (dim 1)
            sample_metric_errors = diff.mean(dim=1).cpu().numpy()
            metric_errors.extend(sample_metric_errors)

    return np.array(errors, dtype=np.float32), np.array(metric_errors, dtype=np.float32)


def train_autoencoder(
    data_dir: Path,
    output_file: Path,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3
) -> dict[str, Any]:
    """Trains the LSTM Autoencoder, calibrates anomaly threshold, and evaluates test splits.

    Workflow:
    1. Loads train/validation/test partitions from `data_dir`.
    2. Calculates baseline statistics (medians, mean, std) on normal training telemetry.
    3. Fits the LSTM Autoencoder using AdamW optimizer and gradient clipping.
    4. Evaluates validation reconstruction errors and determines the F1-optimal threshold.
    5. Evaluates held-out test windows, reporting Precision, Recall, F1, PR-AUC, and FAR.
    6. Persists PyTorch checkpoint with model weights, features, and normalization parameters.

    Args:
        data_dir: Directory containing `train.npz`, `val.npz`, `test.npz`, and `manifest.json`.
        output_file: File destination to save the `.pt` model checkpoint.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: AdamW learning rate.

    Returns:
        dict[str, Any]: Evaluation summary report with training loss and test metrics.
    """
    manifest_file = data_dir / "manifest.json"
    features: list[str] = []
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        features = manifest.get("features", [])

    with np.load(data_dir / "train.npz", allow_pickle=False) as d:
        train_x, train_y = d["X"].astype(np.float32), d["y"]
    with np.load(data_dir / "val.npz", allow_pickle=False) as d:
        val_x, val_y = d["X"].astype(np.float32), d["y"]

    if not len(train_x) or not len(val_x):
        raise ValueError("Training and validation data partitions must contain valid windows.")

    # Calculate statistics from normal training windows (y == 0) to avoid fitting on anomalies
    normal_train = train_x[train_y == 0] if (train_y == 0).sum() > 50 else train_x
    medians = np.nanmedian(normal_train, axis=(0, 1))
    medians = np.nan_to_num(medians, nan=0.0)

    filled_train = np.where(np.isfinite(normal_train), normal_train, medians[None, None, :])
    mean = filled_train.mean((0, 1), keepdims=True)
    std = filled_train.std((0, 1), keepdims=True)
    std[std < 1e-6] = 1.0

    X_train_norm = _normalize_windows(train_x, medians, mean, std)
    X_val_norm = _normalize_windows(val_x, medians, mean, std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = train_x.shape[-1]
    model = LSTMAutoencoder(n_features).to(device)

    # Train only on normal baseline windows if sufficient volume exists
    train_subset = X_train_norm[train_y == 0] if (train_y == 0).sum() > 100 else X_train_norm
    loader = DataLoader(TensorDataset(torch.from_numpy(train_subset)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))

    # Compute validation reconstruction errors
    val_scores, _ = compute_reconstruction_errors(model, X_val_norm, device)

    # Calibrate decision threshold on validation set (maximizing F1 score)
    if len(np.unique(val_y)) == 2:
        p, r, thresholds = precision_recall_curve(val_y, val_scores)
        f1_vals = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
        threshold = float(thresholds[int(np.argmax(f1_vals))])
    else:
        threshold = float(np.quantile(val_scores, 0.95))

    val_pred = (val_scores >= threshold).astype(int)
    val_p = float(precision_score(val_y, val_pred, zero_division=0.0))
    val_r = float(recall_score(val_y, val_pred, zero_division=0.0))
    val_f1 = float(f1_score(val_y, val_pred, zero_division=0.0))
    val_pr_auc = float(average_precision_score(val_y, val_scores)) if len(np.unique(val_y)) > 1 else None
    far = float(val_pred[val_y == 0].mean()) if (val_y == 0).sum() > 0 else 0.0

    report: dict[str, Any] = {
        "model": "LSTM Autoencoder",
        "device": str(device),
        "epochs": epochs,
        "final_train_loss": round(losses[-1], 6),
        "threshold": threshold,
        "validation_metrics": {
            "precision": round(val_p, 4),
            "recall": round(val_r, 4),
            "f1": round(val_f1, 4),
            "pr_auc": round(val_pr_auc, 4) if val_pr_auc is not None else None,
            "false_alarm_rate": round(far, 4),
            "total_windows": len(val_y)
        }
    }

    # Evaluate on held-out test set
    test_file = data_dir / "test.npz"
    if test_file.exists():
        with np.load(test_file, allow_pickle=False) as d:
            test_x, test_y = d["X"].astype(np.float32), d["y"]
        if len(test_x):
            X_test_norm = _normalize_windows(test_x, medians, mean, std)
            test_scores, _ = compute_reconstruction_errors(model, X_test_norm, device)
            test_pred = (test_scores >= threshold).astype(int)
            test_p = float(precision_score(test_y, test_pred, zero_division=0.0))
            test_r = float(recall_score(test_y, test_pred, zero_division=0.0))
            test_f1 = float(f1_score(test_y, test_pred, zero_division=0.0))
            test_pr_auc = float(average_precision_score(test_y, test_scores)) if len(np.unique(test_y)) > 1 else None
            test_far = float(test_pred[test_y == 0].mean()) if (test_y == 0).sum() > 0 else 0.0
            report["test_metrics"] = {
                "precision": round(test_p, 4),
                "recall": round(test_r, 4),
                "f1": round(test_f1, 4),
                "pr_auc": round(test_pr_auc, 4) if test_pr_auc is not None else None,
                "false_alarm_rate": round(test_far, 4),
                "total_windows": len(test_y)
            }

    # Save model checkpoint
    output_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "n_features": n_features,
        "features": features,
        "mean": mean.squeeze().tolist(),
        "std": std.squeeze().tolist(),
        "train_medians": medians.tolist(),
        "threshold": threshold,
        "losses": losses,
        "report": report
    }
    torch.save(checkpoint, output_file)
    (output_file.parent / "lstm_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
