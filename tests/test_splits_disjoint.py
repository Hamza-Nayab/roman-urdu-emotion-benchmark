"""
Unit tests asserting train/val/test global_id sets are strictly pairwise disjoint.
"""

import pytest
import pandas as pd
from romanurdu.splits import load_split_ids, create_canonical_splits


def test_splits_disjoint_mock(tmp_path):
    """Test disjointness assertion logic on synthetic splits."""
    split_dir = tmp_path / "splits"
    split_dir.mkdir()

    # Create synthetic splits matching canonical counts
    train_ids = [f"train_{i:06d}" for i in range(42493)]
    val_ids = [f"val_{i:06d}" for i in range(5312)]
    test_ids = [f"test_{i:06d}" for i in range(5312)]

    pd.DataFrame({"global_id": train_ids}).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame({"global_id": val_ids}).to_csv(split_dir / "val.csv", index=False)
    pd.DataFrame({"global_id": test_ids}).to_csv(split_dir / "test.csv", index=False)

    tr, va, te = load_split_ids(split_dir)
    assert len(tr & va) == 0
    assert len(tr & te) == 0
    assert len(va & te) == 0


def test_splits_leakage_detected(tmp_path):
    """Ensure load_split_ids raises ValueError if leakage is introduced."""
    split_dir = tmp_path / "leaky_splits"
    split_dir.mkdir()

    # Create overlapping splits
    train_ids = [f"id_{i:06d}" for i in range(42493)]
    val_ids = [f"id_{i:06d}" for i in range(42492, 42492 + 5312)]  # 1 overlapping ID
    test_ids = [f"test_{i:06d}" for i in range(5312)]

    pd.DataFrame({"global_id": train_ids}).to_csv(split_dir / "train.csv", index=False)
    pd.DataFrame({"global_id": val_ids}).to_csv(split_dir / "val.csv", index=False)
    pd.DataFrame({"global_id": test_ids}).to_csv(split_dir / "test.csv", index=False)

    with pytest.raises(ValueError, match="Split leakage detected"):
        load_split_ids(split_dir)
