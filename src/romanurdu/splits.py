"""
Deterministic split-freezing and split loading for Roman Urdu emotion benchmark.

Enforces canonical row counts (42,493 train / 5,312 val / 5,312 test) and
guarantees pairwise disjointness by global_id.
"""

from pathlib import Path
from typing import Dict, Set, Tuple, Optional
import pandas as pd

CANONICAL_SPLIT_COUNTS: Dict[str, int] = {
    "train": 42493,
    "val": 5312,
    "test": 5312,
}
EXCLUDED_ROW_IDS: Set[str] = {"validation_004943"}


def load_split_ids(split_dir: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Load frozen train, val, and test global_id sets from split directory.

    Strictly validates:
    1. All three files (train.csv, val.csv, test.csv) exist.
    2. Sets are strictly pairwise disjoint (zero leakage).
    3. Sizes strictly match canonical counts (42,493 / 5,312 / 5,312).

    Args:
        split_dir: Directory containing train.csv, val.csv, and test.csv.

    Returns:
        Tuple of (train_ids, val_ids, test_ids).
    """
    split_dir = Path(split_dir)
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"

    for p in [train_path, val_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Split file missing: {p}")

    train_df = pd.read_csv(train_path, dtype={"global_id": str})
    val_df = pd.read_csv(val_path, dtype={"global_id": str})
    test_df = pd.read_csv(test_path, dtype={"global_id": str})

    train_ids = set(train_df["global_id"].astype(str))
    val_ids = set(val_df["global_id"].astype(str))
    test_ids = set(test_df["global_id"].astype(str))

    # Assert pairwise disjointness
    leak_tr_va = len(train_ids & val_ids)
    leak_tr_te = len(train_ids & test_ids)
    leak_va_te = len(val_ids & test_ids)

    if leak_tr_va != 0 or leak_tr_te != 0 or leak_va_te != 0:
        raise ValueError(
            f"Split leakage detected! tr ∩ va: {leak_tr_va}, "
            f"tr ∩ te: {leak_tr_te}, va ∩ te: {leak_va_te}"
        )

    # Assert canonical sizes
    if len(train_ids) != CANONICAL_SPLIT_COUNTS["train"]:
        raise ValueError(
            f"Train split size mismatch: expected {CANONICAL_SPLIT_COUNTS['train']}, "
            f"found {len(train_ids)}"
        )
    if len(val_ids) != CANONICAL_SPLIT_COUNTS["val"]:
        raise ValueError(
            f"Val split size mismatch: expected {CANONICAL_SPLIT_COUNTS['val']}, "
            f"found {len(val_ids)}"
        )
    if len(test_ids) != CANONICAL_SPLIT_COUNTS["test"]:
        raise ValueError(
            f"Test split size mismatch: expected {CANONICAL_SPLIT_COUNTS['test']}, "
            f"found {len(test_ids)}"
        )

    return train_ids, val_ids, test_ids


def create_canonical_splits(
    df_source: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create canonical frozen splits from source dataframe and optionally persist them.
    Filters out legacy erroneous rows such as 'validation_004943' to guarantee exact
    canonical split populations.
    """
    df = df_source.copy()
    df["global_id"] = df["global_id"].astype(str)

    # Exclude defective extra row if present
    df = df[~df["global_id"].isin(EXCLUDED_ROW_IDS)].reset_index(drop=True)

    # Filter by split if split column exists
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"].isin(["val", "validation"])].copy()
        test_df = df[df["split"] == "test"].copy()
    else:
        # Fallback based on global_id prefix
        train_df = df[df["global_id"].str.startswith("train_")].copy()
        val_df = df[df["global_id"].str.startswith("val")].copy()
        test_df = df[df["global_id"].str.startswith("test_")].copy()

    train_ids = set(train_df["global_id"])
    val_ids = set(val_df["global_id"])
    test_ids = set(test_df["global_id"])

    # Disjointness assertion
    assert len(train_ids & val_ids) == 0, "Train and Val overlap"
    assert len(train_ids & test_ids) == 0, "Train and Test overlap"
    assert len(val_ids & test_ids) == 0, "Val and Test overlap"

    # Canonical count assertions
    assert len(train_ids) == CANONICAL_SPLIT_COUNTS["train"], (
        f"Expected {CANONICAL_SPLIT_COUNTS['train']} train instances, got {len(train_ids)}"
    )
    assert len(val_ids) == CANONICAL_SPLIT_COUNTS["val"], (
        f"Expected {CANONICAL_SPLIT_COUNTS['val']} val instances, got {len(val_ids)}"
    )
    assert len(test_ids) == CANONICAL_SPLIT_COUNTS["test"], (
        f"Expected {CANONICAL_SPLIT_COUNTS['test']} test instances, got {len(test_ids)}"
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_df[["global_id"]].drop_duplicates().to_csv(out / "train.csv", index=False)
        val_df[["global_id"]].drop_duplicates().to_csv(out / "val.csv", index=False)
        test_df[["global_id"]].drop_duplicates().to_csv(out / "test.csv", index=False)

    return train_df, val_df, test_df
