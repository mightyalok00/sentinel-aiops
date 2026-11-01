"""Unit tests for ml_engine/rca_evaluator.py."""

from pathlib import Path

import pandas as pd

from ml_engine.rca_evaluator import evaluate_rca_benchmark, extract_ground_truth


def test_extract_ground_truth():
    """Verifies that service names and fault types are correctly extracted."""
    svc, fault = extract_ground_truth("re1ob_adservice_cpu_1")
    assert svc == "adservice"
    assert fault == "cpu"

    svc2, fault2 = extract_ground_truth("re2tt_frontend_delay_2")
    assert svc2 == "frontend"
    assert fault2 == "delay"

    svc3, fault3 = extract_ground_truth("unknown")
    assert svc3 == "unknown"
    assert fault3 == "unknown"


def test_evaluate_rca_benchmark(tmp_path: Path, monkeypatch):
    """Verifies benchmark calculation with mock incident cases."""
    data_dir = tmp_path / "rcaeval"
    output_file = tmp_path / "rca_results.json"

    case1 = data_dir / "re1ob_adservice_cpu_1"
    case1.mkdir(parents=True)
    (case1 / "inject_time.txt").write_text("1600000000", encoding="utf-8")

    # Generate telemetry with anomalous adservice
    records = [
        {"time": 1600000000.0 + i, "adservice_cpu": 0.95, "cartservice_cpu": 0.10}
        for i in range(40)
    ]
    pd.DataFrame(records).to_parquet(case1 / "metrics.parquet")

    # Mock score_rows to return adservice as top suspect
    def mock_score_rows(rows, model_type="lstm"):
        return {
            "root_cause_analysis": {
                "suspect_services": [
                    {"service": "adservice", "attribution_score": 10.0, "attribution_share_pct": 80.0},
                    {"service": "cartservice", "attribution_score": 2.0, "attribution_share_pct": 20.0}
                ]
            }
        }

    monkeypatch.setattr("ml_engine.rca_evaluator.score_rows", mock_score_rows)

    results = evaluate_rca_benchmark(data_dir, output_path=output_file, max_cases=5)
    assert results["total_cases_evaluated"] == 1
    assert results["hit_at_1_accuracy_pct"] == 100.0
    assert results["hit_at_3_accuracy_pct"] == 100.0
    assert results["mean_reciprocal_rank_mrr"] == 1.0
    assert output_file.exists()
