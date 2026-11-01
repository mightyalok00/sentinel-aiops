"""Unit Tests for RCAEval Telemetry Ingestion, Case Discovery, and Leak-Free Windowing."""

from pathlib import Path

import pytest

from ml_engine.data import discover_cases, read_metrics


def test_discover_cases_empty_directory(tmp_path: Path):
    """Verifies that discover_cases gracefully returns an empty list for a folder without cases."""
    assert discover_cases(tmp_path) == []


def test_discover_cases_pattern_filtering(tmp_path: Path):
    """Verifies that pattern filtering correctly isolates matching cases."""
    case1 = tmp_path / "re1ob_adservice_cpu_1"
    case1.mkdir()
    (case1 / "metrics.parquet").write_bytes(b"")

    case2 = tmp_path / "re1ss_orders_cpu_1"
    case2.mkdir()
    (case2 / "metrics.parquet").write_bytes(b"")

    matching = discover_cases(tmp_path, pattern="re1ob*")
    assert len(matching) == 1
    assert matching[0].name == "re1ob_adservice_cpu_1"


def test_read_metrics_missing_file_raises(tmp_path: Path):
    """Verifies that attempting to read metrics from an empty case folder raises FileNotFoundError."""
    case = tmp_path / "empty_case"
    case.mkdir()
    with pytest.raises(FileNotFoundError):
        read_metrics(case)
