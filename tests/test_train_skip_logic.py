"""
Unit tests for the idempotent execution and skip logic in training pipeline.
"""

from pathlib import Path
from romanurdu.train import is_run_completed


def test_is_run_completed_empty_dir(tmp_path):
    """An empty run directory should not be considered completed."""
    run_dir = tmp_path / "run_01"
    run_dir.mkdir()
    assert is_run_completed(run_dir) is False


def test_is_run_completed_partial_artifacts(tmp_path):
    """A run directory missing any of the 4 required artifacts is not completed."""
    run_dir = tmp_path / "run_02"
    run_dir.mkdir()
    (run_dir / "val_metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "test_metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "classification_report.txt").write_text("report", encoding="utf-8")
    # Missing confusion_matrix.csv
    assert is_run_completed(run_dir) is False


def test_is_run_completed_all_artifacts_present(tmp_path):
    """When all 4 required artifacts exist, is_run_completed returns True."""
    run_dir = tmp_path / "run_03"
    run_dir.mkdir()
    (run_dir / "val_metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "test_metrics.json").write_text("{}", encoding="utf-8")
    (run_dir / "classification_report.txt").write_text("report", encoding="utf-8")
    (run_dir / "confusion_matrix.csv").write_text("1,0\n0,1", encoding="utf-8")
    assert is_run_completed(run_dir) is True
