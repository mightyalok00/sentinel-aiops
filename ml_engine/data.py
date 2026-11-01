"""RCAEval Dataset Ingestion, Inspection, and Windowing Pipeline.

This module provides data loading and preprocessing routines:
1. Downloader: Fetches specific benchmark suites (RE1, RE2, RE3) from Hugging Face Hub.
2. Case Discovery: Recursively identifies incident cases adhering to Parquet/JSON schemas.
3. Incident-Level Partitioning: Splits full incident case directories into train, validation,
   and test splits BEFORE windowing. This guarantees zero window leakage from the same
   incident across splits.
4. Window Generation: Slices multi-variate telemetry series into fixed-length windows with
   user-configurable stride and injection-time proxy labels.
"""

import fnmatch
from pathlib import Path
import json
from typing import Any, Optional
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.model_selection import train_test_split


def download_dataset(suite: str, output_dir: Path) -> Path:
    """Downloads a targeted RCAEval benchmark suite from Hugging Face Hub.

    Args:
        suite: Target suite identifier ('re1', 're2', or 're3').
        output_dir: Local directory path to store downloaded dataset files.

    Returns:
        Path: Output directory path containing the downloaded repository snapshot.

    Raises:
        ValueError: If `suite` is not one of {'re1', 're2', 're3'}.
    """
    suite = suite.lower()
    if suite not in {"re1", "re2", "re3"}:
        raise ValueError(f"Invalid suite '{suite}'. Must be 're1', 're2', or 're3'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="phamquiluan/RCAEval",
        repo_type="dataset",
        allow_patterns=[f"{suite}*", "cases.parquet"],
        local_dir=str(output_dir),
    )
    return output_dir


def discover_cases(data_dir: Path, pattern: Optional[str] = None) -> list[Path]:
    """Scans the dataset directory for directories containing metric telemetry.

    Args:
        data_dir: Root dataset folder containing case subdirectories.
        pattern: Optional glob/fnmatch pattern to filter cases (e.g. 're1ob*').

    Returns:
        list[Path]: Sorted list of valid case directory paths.

    Raises:
        FileNotFoundError: If `data_dir` does not exist.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    folders = [
        p
        for p in data_dir.iterdir()
        if p.is_dir()
        and ((p / "metrics.parquet").exists() or (p / "metrics.json").exists())
    ]
    if pattern:
        folders = [p for p in folders if fnmatch.fnmatch(p.name, pattern)]
    return sorted(folders)


def read_metrics(case_dir: Path) -> pd.DataFrame:
    """Reads and sanitizes metric time-series from parquet or JSON files.

    Converts time to numeric seconds, sorts rows chronologically, replaces infinities
    with NaNs, and drops rows with invalid timestamps.

    Args:
        case_dir: Directory path for an individual incident case.

    Returns:
        pd.DataFrame: Sanitized time-series dataframe with sorted 'time' column.

    Raises:
        FileNotFoundError: If neither metrics.parquet nor metrics.json exists.
        ValueError: If 'time' column is missing from the dataset.
    """
    parquet = case_dir / "metrics.parquet"
    json_file = case_dir / "metrics.json"

    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif json_file.exists():
        df = pd.DataFrame(json.loads(json_file.read_text(encoding="utf-8")))
    else:
        raise FileNotFoundError(f"No metrics file found in {case_dir}")

    if "time" not in df.columns:
        raise ValueError(f"Missing required 'time' column in {case_dir}")

    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    return (
        df.dropna(subset=["time"])
        .sort_values("time")
        .replace([np.inf, -np.inf], np.nan)
    )


def inspect_dataset(data_dir: Path) -> dict[str, Any]:
    """Surveys dataset integrity, schema dimensions, and multi-signal coverage.

    Args:
        data_dir: Dataset root directory path.

    Returns:
        dict[str, Any]: Case count, per-case summaries (metrics count, logs, traces), and parsing errors.
    """
    cases: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for folder in discover_cases(data_dir):
        try:
            frame = read_metrics(folder)
            cases.append({
                "case": folder.name,
                "rows": len(frame),
                "numeric_metrics": len(frame.select_dtypes("number").columns) - 1,
                "has_logs": (folder / "logs.parquet").exists() or (folder / "logs.csv").exists(),
                "has_traces": (folder / "traces.parquet").exists() or (folder / "traces.csv").exists(),
            })
        except Exception as exc:
            errors.append({"case": folder.name, "error": str(exc)})

    return {"case_count": len(cases), "cases": cases, "errors": errors}


def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    window_size: int = 30,
    stride: int = 5,
    pattern: str = "re1ob*"
) -> dict[str, Any]:
    """Generates train/validation/test temporal windows with zero cross-incident leakage.

    Split Architecture:
    1. Incident Discovery: Groups cases by system topology via `pattern`.
    2. Case-Level Partitioning: Complete incident directories are divided:
       - 60% Train
       - 20% Validation
       - 20% Held-out Test
       This prevents contiguous sliding windows of the same incident from appearing in both train and test.
    3. Window Extraction: Slices each incident into (window_size, n_features) arrays with `stride`.
    4. Proxy Labeling: Time steps occurring at or after `inject_time.txt` are labeled 1 (anomalous).
    5. Persistence: Saves compressed `.npz` arrays and `manifest.json`.

    Args:
        data_dir: Source directory containing raw RCAEval incident cases.
        output_dir: Destination directory for processed `.npz` partitions.
        window_size: Number of consecutive timestamps per window (default: 30).
        stride: Step size between consecutive sliding windows (default: 5).
        pattern: Case name filter ensuring homogeneous microservice metric schema (default: 're1ob*').

    Returns:
        dict[str, Any]: Manifest summary with feature schema, window counts, and positive label rates.

    Raises:
        ValueError: If parameters are invalid or fewer than 3 incident cases are available.
    """
    if window_size < 2 or stride < 1:
        raise ValueError(f"Invalid window parameters: window_size={window_size} (must >= 2), stride={stride} (must >= 1).")

    case_dirs = discover_cases(data_dir, pattern=pattern)
    if len(case_dirs) < 3:
        # Fallback to all discovered cases if pattern matched too few
        case_dirs = discover_cases(data_dir)
    if len(case_dirs) < 3:
        raise ValueError(f"At least 3 incident case directories required for 3-way split; found {len(case_dirs)}.")

    # Partition incident IDs before sliding windows are formed
    ids = np.arange(len(case_dirs))
    train_val, test = train_test_split(ids, test_size=0.20, random_state=42)
    train, val = train_test_split(train_val, test_size=0.25, random_state=43)

    split = {case_dirs[i].name: "train" for i in train}
    split.update({case_dirs[i].name: "val" for i in val})
    split.update({case_dirs[i].name: "test" for i in test})

    collected: dict[str, list[Any]] = {s: [] for s in ("train", "val", "test")}
    skipped: list[dict[str, str]] = []
    feature_names: Optional[list[str]] = None

    for folder in case_dirs:
        try:
            frame = read_metrics(folder)
            numeric = frame.select_dtypes("number").drop(columns=["time"], errors="ignore")
            if numeric.empty:
                raise ValueError("No numeric metric columns available in case.")

            if feature_names is None:
                feature_names = sorted(list(numeric.columns))

            # Guarantee column schema alignment across all cases
            numeric = numeric.reindex(columns=feature_names, fill_value=np.nan)
            x = numeric.to_numpy(dtype=np.float32, copy=True)
            x[~np.isfinite(x)] = np.nan

            injection_file = folder / "inject_time.txt"
            if not injection_file.exists():
                raise FileNotFoundError("inject_time.txt missing in incident folder.")
            injection_time = float(injection_file.read_text(encoding="utf-8").strip())

            times = frame["time"].to_numpy(dtype=float)
            for start in range(0, len(x) - window_size + 1, stride):
                end = start + window_size
                if np.isnan(x[start:end]).all():
                    continue
                # Label is 1 if the final window timestamp falls into or after the fault injection
                label = int(times[end - 1] >= injection_time)
                collected[split[folder.name]].append((x[start:end], label, folder.name))
        except Exception as exc:
            skipped.append({"case": folder.name, "error": str(exc)})

    if not feature_names:
        raise ValueError("No usable metric features found across target incident cases.")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "features": feature_names,
        "window_size": window_size,
        "stride": stride,
        "label_note": "Post-injection proxy labels; not independently adjudicated.",
        "case_counts": {s: len({r[2] for r in collected[s]}) for s in collected},
        "skipped": skipped,
    }

    for part, records in collected.items():
        if records:
            X = np.stack([r[0] for r in records]).astype(np.float32)
            y = np.asarray([r[1] for r in records], dtype=np.int8)
            groups = np.asarray([r[2] for r in records])
        else:
            X = np.empty((0, window_size, len(feature_names)), dtype=np.float32)
            y = np.empty((0,), dtype=np.int8)
            groups = np.empty((0,), dtype=str)

        np.savez_compressed(output_dir / f"{part}.npz", X=X, y=y, groups=groups)
        manifest[f"{part}_windows"] = int(len(y))
        manifest[f"{part}_positive_rate"] = float(y.mean()) if len(y) else None

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
