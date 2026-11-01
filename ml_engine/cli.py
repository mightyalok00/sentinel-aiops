"""Command Line Interface (CLI) for SENTINEL AI.

Provides developer workflows for:
- Downloading RCAEval benchmark suites from Hugging Face Hub
- Inspecting local case directories and telemetry modalities
- Splitting incident cases into leak-free train/validation/test partitions
- Training Isolation Forest baseline and PyTorch LSTM Autoencoder models
- Evaluating and comparing model benchmarks on held-out test telemetry
"""

import argparse
import json
from pathlib import Path
from typing import Any

from ml_engine.checksums import generate_dataset_checksums, verify_dataset_checksums
from ml_engine.data import download_dataset, inspect_dataset, prepare_dataset
from ml_engine.lstm_autoencoder import train_autoencoder
from ml_engine.rca_evaluator import evaluate_rca_benchmark
from ml_engine.train import train_isolation_forest


def evaluate_models(data_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    """Collates and formats evaluation reports from all trained models in `artifacts_dir`.

    Args:
        data_dir: Processed dataset directory path.
        artifacts_dir: Artifacts directory containing `evaluation.json` and `lstm_evaluation.json`.

    Returns:
        dict[str, Any]: Dictionary containing status and benchmark summaries per model.
    """
    results: dict[str, Any] = {}
    iforest_eval = artifacts_dir / "evaluation.json"
    if iforest_eval.exists():
        results["isolation_forest"] = json.loads(iforest_eval.read_text(encoding="utf-8"))
    lstm_eval = artifacts_dir / "lstm_evaluation.json"
    if lstm_eval.exists():
        results["lstm_autoencoder"] = json.loads(lstm_eval.read_text(encoding="utf-8"))
    return {
        "status": "success",
        "benchmark_summary": results,
    }


def main() -> None:
    """Parses command-line arguments and dispatches corresponding subcommands."""
    parser = argparse.ArgumentParser(
        description="SENTINEL AI: AIOps Telemetry Anomaly Scoring & Root Cause Analysis CLI"
    )
    sub = parser.add_subparsers(dest="cmd", required=True, help="Subcommand to execute")

    # Command: download
    p_dl = sub.add_parser("download", help="Download RCAEval suite from Hugging Face Hub")
    p_dl.add_argument("--suite", default="re1", choices=["re1", "re2", "re3"], help="RCAEval suite ID")
    p_dl.add_argument("--output", type=Path, default=Path("data/rcaeval"), help="Download output folder")

    # Command: inspect
    p_ins = sub.add_parser("inspect", help="Inspect schema and modality coverage of dataset")
    p_ins.add_argument("--data-dir", type=Path, default=Path("data/rcaeval"), help="Dataset directory")

    # Command: prepare
    p_prep = sub.add_parser("prepare", help="Window and partition incident cases into train/val/test")
    p_prep.add_argument("--data-dir", type=Path, default=Path("data/rcaeval"), help="Raw dataset folder")
    p_prep.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Output directory")
    p_prep.add_argument("--pattern", default="re1ob*", help="Glob pattern to select homogeneous system cases")
    p_prep.add_argument("--window-size", type=int, default=30, help="Timestamps per window")
    p_prep.add_argument("--stride", type=int, default=5, help="Sliding step between windows")

    # Command: train-iforest
    p_tr_if = sub.add_parser("train-iforest", help="Train Isolation Forest statistical baseline")
    p_tr_if.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed data folder")
    p_tr_if.add_argument("--output-dir", type=Path, default=Path("artifacts"), help="Artifact destination folder")

    # Command: train-lstm
    p_tr_lstm = sub.add_parser("train-lstm", help="Train PyTorch LSTM Autoencoder on temporal sequences")
    p_tr_lstm.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed data folder")
    p_tr_lstm.add_argument("--output-file", type=Path, default=Path("artifacts/lstm_autoencoder.pt"), help="Checkpoint path")
    p_tr_lstm.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    p_tr_lstm.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    p_tr_lstm.add_argument("--lr", type=float, default=1e-3, help="Learning rate for AdamW")

    # Command: evaluate
    p_eval = sub.add_parser("evaluate", help="Collate and display side-by-side benchmark comparison")
    p_eval.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Processed data folder")
    p_eval.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"), help="Artifacts directory")

    # Command: evaluate-rca
    p_rca = sub.add_parser("evaluate-rca", help="Benchmark empirical Top-K Root Cause Analysis accuracy (Hit@1, Hit@3, MRR)")
    p_rca.add_argument("--data-dir", type=Path, default=Path("data/rcaeval"), help="Raw benchmark dataset folder")
    p_rca.add_argument("--model", default="lstm", choices=["lstm", "iforest"], help="Model engine to evaluate")
    p_rca.add_argument("--pattern", default="re1ob*", help="Case glob filter pattern")
    p_rca.add_argument("--max-cases", type=int, default=None, help="Optional limit on cases to evaluate")
    p_rca.add_argument("--output", type=Path, default=Path("artifacts/rca_benchmark_results.json"), help="Output path")

    # Command: checksums
    p_chk = sub.add_parser("checksums", help="Generate SHA-256 reproducibility checksums manifest for dataset")
    p_chk.add_argument("--data-dir", type=Path, default=Path("data/rcaeval"), help="Raw benchmark dataset folder")
    p_chk.add_argument("--output", type=Path, default=Path("data/checksums.json"), help="Manifest destination")
    p_chk.add_argument("--pattern", default="re1ob*", help="Case glob filter pattern")

    # Command: verify-checksums
    p_vchk = sub.add_parser("verify-checksums", help="Verify dataset integrity against SHA-256 checksums manifest")
    p_vchk.add_argument("--data-dir", type=Path, default=Path("data/rcaeval"), help="Raw benchmark dataset folder")
    p_vchk.add_argument("--manifest", type=Path, default=Path("data/checksums.json"), help="Manifest path to verify")

    args = parser.parse_args()

    if args.cmd == "download":
        result = str(download_dataset(args.suite, args.output))
    elif args.cmd == "inspect":
        result = inspect_dataset(args.data_dir)
    elif args.cmd == "prepare":
        result = prepare_dataset(args.data_dir, args.output_dir, args.window_size, args.stride, args.pattern)
    elif args.cmd == "train-iforest":
        result = train_isolation_forest(args.data_dir, args.output_dir)
    elif args.cmd == "train-lstm":
        result = train_autoencoder(args.data_dir, args.output_file, args.epochs, args.batch_size, args.lr)
    elif args.cmd == "evaluate":
        result = evaluate_models(args.data_dir, args.artifacts_dir)
    elif args.cmd == "evaluate-rca":
        result = evaluate_rca_benchmark(args.data_dir, args.model, args.pattern, args.max_cases, args.output)
    elif args.cmd == "checksums":
        result = generate_dataset_checksums(args.data_dir, args.output, args.pattern)
    elif args.cmd == "verify-checksums":
        result = verify_dataset_checksums(args.data_dir, args.manifest)
    else:
        result = {"error": f"Unknown subcommand: {args.cmd}"}

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
