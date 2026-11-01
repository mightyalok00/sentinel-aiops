"""Unit tests for ml_engine/cli.py argument parsing and commands."""

from pathlib import Path

from ml_engine.cli import build_parser, evaluate_models


def test_cli_parser_prepare():
    """Verifies prepare subcommand argument parsing."""
    parser = build_parser()
    args = parser.parse_args(["prepare", "--data-dir", "data/rcaeval", "--output-dir", "data/processed", "--pattern", "re1ob*"])
    assert args.cmd == "prepare"
    assert str(args.data_dir).replace("\\", "/") == "data/rcaeval"
    assert str(args.output_dir).replace("\\", "/") == "data/processed"
    assert args.pattern == "re1ob*"


def test_cli_parser_train_iforest():
    """Verifies train-iforest subcommand argument parsing."""
    parser = build_parser()
    args = parser.parse_args(["train-iforest", "--data-dir", "data/processed", "--output-dir", "artifacts"])
    assert args.cmd == "train-iforest"
    assert str(args.data_dir).replace("\\", "/") == "data/processed"
    assert str(args.output_dir).replace("\\", "/") == "artifacts"


def test_cli_parser_train_lstm():
    """Verifies train-lstm subcommand argument parsing."""
    parser = build_parser()
    args = parser.parse_args(["train-lstm", "--epochs", "5", "--batch-size", "16"])
    assert args.cmd == "train-lstm"
    assert args.epochs == 5
    assert args.batch_size == 16


def test_cli_evaluate_models_empty(tmp_path: Path):
    """Verifies evaluate_models handles empty or missing directories gracefully."""
    res = evaluate_models(tmp_path, tmp_path)
    assert res["status"] == "success"
    assert res["benchmark_summary"] == {}
