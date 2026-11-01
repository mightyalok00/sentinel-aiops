"""Empirical Root Cause Analysis (RCA) Benchmark Evaluator.

Evaluates microservice fault localization performance against RCAEval ground-truth
fault injection targets. Computes industry-standard AIOps localization metrics:
- Hit@1 (Top-1 Accuracy): Target service is ranked as the #1 most suspect service.
- Hit@3 (Top-3 Accuracy): Target service is within the top 3 suspect services.
- Hit@5 (Top-5 Accuracy): Target service is within the top 5 suspect services.
- MRR (Mean Reciprocal Rank): 1 / rank of the target service.

Also segments accuracy by fault type (e.g. CPU saturation, network delay, packet loss,
memory leak, disk I/O pressure).
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.scoring import score_rows


def extract_ground_truth(case_folder_name: str) -> tuple[str, str]:
    """Extracts target faulty service and fault category from case directory name.

    Example:
        're1ob_adservice_cpu_1' -> ('adservice', 'cpu')
        're1ob_cartservice_delay_2' -> ('cartservice', 'delay')
    """
    parts = case_folder_name.split("_")
    if len(parts) >= 3:
        target_service = parts[1]
        fault_type = parts[2]
        return target_service, fault_type
    return "unknown", "unknown"


def evaluate_rca_benchmark(
    data_dir: Path,
    model_type: str = "lstm",
    pattern: str = "re1ob*",
    max_cases: Optional[int] = None,
    output_path: Optional[Path] = None
) -> dict[str, Any]:
    """Evaluates Top-K root cause attribution accuracy across benchmark cases.

    Args:
        data_dir: Path to RCAEval benchmark dataset directory.
        model_type: Model engine to evaluate ('lstm' or 'iforest').
        pattern: Glob pattern to filter benchmark cases (e.g. 're1ob*').
        max_cases: Optional limit on the number of cases to evaluate for quick runs.
        output_path: Optional path to save JSON results.

    Returns:
        dict[str, Any]: Aggregated benchmark metrics including Hit@1, Hit@3, Hit@5, MRR,
        and fault-type breakdown.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    cases = sorted([
        p for p in data_dir.glob(pattern)
        if p.is_dir() and ((p / "metrics.parquet").exists() or (p / "metrics.json").exists())
    ])

    if max_cases and max_cases > 0:
        cases = cases[:max_cases]

    if not cases:
        return {
            "error": f"No cases found matching pattern '{pattern}' in {data_dir}",
            "evaluated_cases": 0
        }

    hits_1: list[int] = []
    hits_3: list[int] = []
    hits_5: list[int] = []
    reciprocal_ranks: list[float] = []

    fault_type_stats: dict[str, dict[str, list[float]]] = {}

    for case_path in cases:
        target_service, fault_type = extract_ground_truth(case_path.name)
        if target_service == "unknown":
            continue

        # Load metrics
        parquet_file = case_path / "metrics.parquet"
        json_file = case_path / "metrics.json"

        try:
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
            elif json_file.exists():
                df = pd.DataFrame(json.loads(json_file.read_text(encoding="utf-8")))
            else:
                continue

            if "time" not in df.columns:
                continue

            df = df.copy()
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time")

            # Check injection time
            inj_file = case_path / "inject_time.txt"
            if inj_file.exists():
                inj_time = float(inj_file.read_text(encoding="utf-8").strip())
                # Isolate the anomaly window around and after injection
                post_inj = df[df["time"] >= inj_time]
                if len(post_inj) >= 10:
                    eval_df = post_inj.head(60)
                else:
                    eval_df = df.tail(60)
            else:
                eval_df = df.tail(60)

            clean_df = eval_df.drop(columns=["time"], errors="ignore")
            rows: list[dict[str, Any]] = [dict(r) for _, r in clean_df.iterrows()]
            if not rows:
                continue

            # Run scoring and attribution
            result = score_rows(rows, model_type=model_type)
            suspects = result.get("root_cause_analysis", {}).get("suspect_services", [])
            predicted = [s["service"].lower() for s in suspects]

            # Calculate Hit@K and MRR
            h1 = int(len(predicted) > 0 and predicted[0] == target_service.lower())
            h3 = int(target_service.lower() in predicted[:3])
            h5 = int(target_service.lower() in predicted[:5])

            if target_service.lower() in predicted:
                rank = predicted.index(target_service.lower()) + 1
                rr = 1.0 / rank
            else:
                rr = 0.0

            hits_1.append(h1)
            hits_3.append(h3)
            hits_5.append(h5)
            reciprocal_ranks.append(rr)

            if fault_type not in fault_type_stats:
                fault_type_stats[fault_type] = {"hit_1": [], "hit_3": [], "mrr": []}
            fault_type_stats[fault_type]["hit_1"].append(h1)
            fault_type_stats[fault_type]["hit_3"].append(h3)
            fault_type_stats[fault_type]["mrr"].append(rr)

        except Exception:
            continue

    total_evaluated = len(hits_1)
    if total_evaluated == 0:
        return {"error": "No cases could be evaluated successfully", "evaluated_cases": 0}

    fault_breakdown = {
        ft: {
            "cases": len(stats["hit_1"]),
            "hit_at_1_pct": round(float(np.mean(stats["hit_1"])) * 100, 1),
            "hit_at_3_pct": round(float(np.mean(stats["hit_3"])) * 100, 1),
            "mrr": round(float(np.mean(stats["mrr"])), 3)
        }
        for ft, stats in fault_type_stats.items()
    }

    summary = {
        "benchmark_suite": pattern,
        "model_architecture": model_type.upper(),
        "total_cases_evaluated": total_evaluated,
        "hit_at_1_accuracy_pct": round(float(np.mean(hits_1)) * 100, 1),
        "hit_at_3_accuracy_pct": round(float(np.mean(hits_3)) * 100, 1),
        "hit_at_5_accuracy_pct": round(float(np.mean(hits_5)) * 100, 1),
        "mean_reciprocal_rank_mrr": round(float(np.mean(reciprocal_ranks)), 3),
        "fault_type_breakdown": fault_breakdown
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
