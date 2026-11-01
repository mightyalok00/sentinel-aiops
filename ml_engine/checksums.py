"""Dataset Checksums & Reproducibility Verification Engine.

Computes and verifies cryptographic SHA-256 checksums across benchmark files
(metrics, logs, traces, and injection timestamps) to guarantee reproducibility,
prevent data corruption, and ensure academic validation integrity.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def compute_file_sha256(file_path: Path) -> str:
    """Computes the SHA-256 cryptographic digest of a file.

    Args:
        file_path: Absolute or relative Path to target file.

    Returns:
        str: Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_dataset_checksums(
    data_dir: Path,
    output_manifest: Optional[Path] = None,
    pattern: str = "re1ob*"
) -> dict[str, Any]:
    """Scans benchmark case directories and generates a SHA-256 verification manifest.

    Args:
        data_dir: Root benchmark dataset directory (e.g. data/rcaeval).
        output_manifest: Optional Path to write the JSON manifest to.
        pattern: Glob pattern to filter cases.

    Returns:
        dict[str, Any]: Manifest containing file paths and their respective SHA-256 hashes.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    cases = sorted([p for p in data_dir.glob(pattern) if p.is_dir()])
    file_hashes: dict[str, str] = {}

    for case_dir in cases:
        for fname in ["metrics.parquet", "metrics.json", "inject_time.txt", "log.txt", "trace.json"]:
            fpath = case_dir / fname
            if fpath.exists() and fpath.is_file():
                rel_path = f"{case_dir.name}/{fname}"
                file_hashes[rel_path] = compute_file_sha256(fpath)

    manifest = {
        "hash_algorithm": "SHA-256",
        "case_pattern": pattern,
        "total_cases_indexed": len(cases),
        "total_files_hashed": len(file_hashes),
        "checksums": file_hashes
    }

    if output_manifest:
        output_manifest.parent.mkdir(parents=True, exist_ok=True)
        output_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def verify_dataset_checksums(
    data_dir: Path,
    manifest_path: Path
) -> dict[str, Any]:
    """Verifies dataset files against an existing SHA-256 checksum manifest.

    Args:
        data_dir: Root benchmark dataset directory.
        manifest_path: Path to the JSON checksum manifest.

    Returns:
        dict[str, Any]: Verification summary detailing verified, corrupted, or missing files.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_checksums: dict[str, str] = manifest.get("checksums", {})

    verified: list[str] = []
    mismatches: list[dict[str, str]] = []
    missing: list[str] = []

    for rel_path, expected_hash in expected_checksums.items():
        target_path = data_dir / rel_path
        if not target_path.exists():
            missing.append(rel_path)
            continue

        actual_hash = compute_file_sha256(target_path)
        if actual_hash == expected_hash:
            verified.append(rel_path)
        else:
            mismatches.append({
                "file": rel_path,
                "expected": expected_hash,
                "actual": actual_hash
            })

    is_valid = len(mismatches) == 0 and len(missing) == 0

    return {
        "valid": is_valid,
        "total_checked": len(expected_checksums),
        "verified_count": len(verified),
        "mismatches": mismatches,
        "missing": missing
    }
