"""Unit Tests for RCAEval Telemetry Ingestion and Case Discovery."""

from pathlib import Path
from ml_engine.data import discover_cases


def test_discover_cases_empty_directory(tmp_path: Path):
    """Verifies that discover_cases gracefully returns an empty list for a folder without cases."""
    assert discover_cases(tmp_path) == []
