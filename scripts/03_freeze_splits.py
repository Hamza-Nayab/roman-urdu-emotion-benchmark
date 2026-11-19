#!/usr/bin/env python3
"""
Step 3: Freeze canonical train, validation, and test splits (42,493 / 5,312 / 5,312).
Strictly verifies pairwise disjointness and eliminates any legacy extra rows.
"""

import argparse
from pathlib import Path
import pandas as pd
from romanurdu.splits import create_canonical_splits, CANONICAL_SPLIT_COUNTS


def main():
    parser = argparse.ArgumentParser(description="Freeze canonical split IDs")
    parser.add_argument("--source-csv", type=str, default="data/goemotions_roman_ur_firstlabel.csv")
    parser.add_argument("--output-dir", type=str, default="data/splits")
    args = parser.parse_args()

    source_path = Path(args.source_csv)
    output_dir = Path(args.output_dir)

    if not source_path.exists():
        # Fallback to English source if translation not run yet
        en_source = Path("data/goemotions_en_all_firstlabel.csv")
        if en_source.exists():
            source_path = en_source
        else:
            raise FileNotFoundError(f"Source file not found at {source_path} or {en_source}")

    print(f"Reading instances from {source_path}...")
    source_df = pd.read_csv(source_path)

    tr, va, te = create_canonical_splits(source_df, output_dir=output_dir)

    print("\nCanonical splits frozen successfully:")
    print(f"  Train: {len(tr)} instances (expected {CANONICAL_SPLIT_COUNTS['train']})")
    print(f"  Val:   {len(va)} instances (expected {CANONICAL_SPLIT_COUNTS['val']})")
    print(f"  Test:  {len(te)} instances (expected {CANONICAL_SPLIT_COUNTS['test']})")
    print(f"  Total: {len(tr) + len(va) + len(te)} instances")
    print(f"Artifacts saved in: {output_dir}")


if __name__ == "__main__":
    main()
