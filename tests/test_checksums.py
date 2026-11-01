"""Tests for dataset SHA-256 reproducibility checksums."""

import json

from ml_engine.checksums import generate_dataset_checksums, verify_dataset_checksums


def test_checksum_manifest_generation_and_verification(tmp_path):
    """Verifies that dataset checksums can be generated and successfully verified."""
    # Create mock case structure
    case_dir = tmp_path / "re1ob_test_cpu_1"
    case_dir.mkdir(parents=True)
    (case_dir / "inject_time.txt").write_text("1600000000", encoding="utf-8")
    (case_dir / "metrics.json").write_text(json.dumps([{"time": 1, "cpu": 0.5}]), encoding="utf-8")

    manifest_path = tmp_path / "checksums.json"
    manifest = generate_dataset_checksums(tmp_path, manifest_path, pattern="re1ob*")

    assert manifest["total_cases_indexed"] == 1
    assert manifest["total_files_hashed"] == 2
    assert manifest_path.exists()

    # Verify checksums
    verification = verify_dataset_checksums(tmp_path, manifest_path)
    assert verification["valid"] is True
    assert verification["verified_count"] == 2
    assert len(verification["mismatches"]) == 0
    assert len(verification["missing"]) == 0


def test_checksum_detects_corrupted_file(tmp_path):
    """Verifies that tampering with a file is immediately caught by verification."""
    case_dir = tmp_path / "re1ob_test_mem_1"
    case_dir.mkdir(parents=True)
    target_file = case_dir / "inject_time.txt"
    target_file.write_text("1000", encoding="utf-8")

    manifest_path = tmp_path / "checksums.json"
    generate_dataset_checksums(tmp_path, manifest_path, pattern="re1ob*")

    # Corrupt file content
    target_file.write_text("9999", encoding="utf-8")

    verification = verify_dataset_checksums(tmp_path, manifest_path)
    assert verification["valid"] is False
    assert len(verification["mismatches"]) == 1
