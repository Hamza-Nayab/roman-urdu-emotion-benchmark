"""
Unit tests asserting canonical split counts (42,493 train / 5,312 val / 5,312 test).
Guarantees that the historical one-example discrepancy (42,494 with validation_004943) cannot regress.
"""

from pathlib import Path
import pytest
import pandas as pd
from romanurdu.splits import (
    CANONICAL_SPLIT_COUNTS,
    EXCLUDED_ROW_IDS,
    load_split_ids,
    create_canonical_splits,
)


def test_split_size_constants():
    """Verify constant definitions match the paper specification exactly."""
    assert CANONICAL_SPLIT_COUNTS["train"] == 42493
    assert CANONICAL_SPLIT_COUNTS["val"] == 5312
    assert CANONICAL_SPLIT_COUNTS["test"] == 5312
    assert sum(CANONICAL_SPLIT_COUNTS.values()) == 53117
    assert "validation_004943" in EXCLUDED_ROW_IDS


def test_create_canonical_splits_strips_defective_row():
    """Verify create_canonical_splits strips validation_004943 and enforces exact sizes."""
    # Synthesize data including the extra row
    train_rows = [{"global_id": f"train_{i:06d}", "split": "train"} for i in range(42493)]
    # Add extra row to train
    train_rows.append({"global_id": "validation_004943", "split": "train"})
    val_rows = [{"global_id": f"val_{i:06d}", "split": "val"} for i in range(5312)]
    test_rows = [{"global_id": f"test_{i:06d}", "split": "test"} for i in range(5312)]

    df = pd.DataFrame(train_rows + val_rows + test_rows)
    tr, va, te = create_canonical_splits(df)

    assert len(tr) == 42493
    assert len(va) == 5312
    assert len(te) == 5312
    assert "validation_004943" not in set(tr["global_id"])


def test_split_size_mismatch_raises(tmp_path):
    """Verify load_split_ids raises if split has 42,494 rows instead of 42,493."""
    split_dir = tmp_path / "bad_splits"
    split_dir.mkdir()

    # 42,494 rows (defective)
    train_ids = [f"train_{i:06d}" for i in range(42494)]
    val_ids = [f"val_{i:06d}" for i in range(5312)]
    test_ids = [f"test_{i:06d}" for i in range(5312)]

    pd.DataFrame({"global_id": train_ids}).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame({"global_id": val_ids}).to_csv(split_dir / "val.csv", index=False)
    pd.DataFrame({"global_id": test_ids}).to_csv(split_dir / "test.csv", index=False)

    with pytest.raises(ValueError, match="Train split size mismatch"):
        load_split_ids(split_dir)


def test_canonical_zenodo_splits_on_disk():
    """Verify released zenodo_release ablation_off splits match canonical counts if present."""
    zenodo_dir = Path(__file__).resolve().parents[3] / "zenodo_release" / "data" / "ablation_off"
    if zenodo_dir.exists():
        tr, va, te = load_split_ids(zenodo_dir)
        assert len(tr) == 42493
        assert len(va) == 5312
        assert len(te) == 5312
