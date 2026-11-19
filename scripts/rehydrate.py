#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rehydrate GoEmotions English text locally and join with this Zenodo Release B package
using global_id.

This Zenodo package does NOT ship English text. This script:
  1) downloads GoEmotions (simplified) from HuggingFace,
  2) recreates global_id = "{split}_{idx:06d}",
  3) joins English text onto the release CSVs,
  4) writes LOCAL merged outputs for verification.

USAGE (run from zenodo_release/):
  pip install -r requirements.txt
  python code/rehydrate.py --release_dir .

Outputs (LOCAL ONLY; do not upload):
  LOCAL_off_merged_with_english.csv
  LOCAL_on_merged_with_english.csv
"""

import argparse
from pathlib import Path
import pandas as pd
from datasets import load_dataset


def build_goemotions_lookup() -> pd.DataFrame:
    """
    Download GoEmotions simplified and build a lookup table:
      global_id -> text_en

    global_id format matches your pipeline:
      split_000000  (idx is the row index after reset_index(drop=True))
    """
    ds = load_dataset("go_emotions", "simplified")

    parts = []
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        df = ds[split].to_pandas().reset_index(drop=True)

        # GoEmotions uses column name "text" for English text
        if "text" not in df.columns:
            raise ValueError(f"Expected column 'text' in GoEmotions {split} split.")

        df["idx"] = df.index.astype(int)
        df["global_id"] = split + "_" + df["idx"].astype(str).str.zfill(6)
        df = df[["global_id", "text"]].rename(columns={"text": "text_en"})
        parts.append(df)

    lookup = pd.concat(parts, ignore_index=True)
    if lookup["global_id"].duplicated().any():
        raise ValueError("Duplicate global_id found in GoEmotions lookup (unexpected).")
    return lookup


def read_release_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path, dtype=str)
    required = {"global_id", "label", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}. Found: {list(df.columns)}")
    df["global_id"] = df["global_id"].astype(str)
    return df


def merge_and_write(view_name: str, view_dir: Path, lookup: pd.DataFrame, out_path: Path) -> None:
    # Load splits
    train = read_release_csv(view_dir / "train.csv")
    val   = read_release_csv(view_dir / "val.csv")
    test  = read_release_csv(view_dir / "test.csv")

    # Combine to one file for convenience
    merged = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")],
        ignore_index=True
    )

    # Join English
    out = merged.merge(lookup, on="global_id", how="left")

    missing_en = out["text_en"].isna().sum()
    total = len(out)

    # Reorder columns nicely
    out = out[["global_id", "split", "label", "text_en", "text"]].rename(columns={"text": "text_roman_ur"})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[{view_name}] Wrote: {out_path}")
    print(f"[{view_name}] Rows: {total}, missing English matches: {missing_en}")

    if missing_en > 0:
        # show a few examples to debug
        print(f"[{view_name}] Example missing global_id(s):")
        print(out.loc[out["text_en"].isna(), "global_id"].head(10).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--release_dir",
        type=str,
        default=".",
        help="Path to zenodo_release directory (use '.' when running from inside it)."
    )
    args = ap.parse_args()

    release_dir = Path(args.release_dir).resolve()
    data_dir = release_dir / "data"

    off_dir = data_dir / "ablation_off"
    on_dir  = data_dir / "ablation_on"

    if not off_dir.exists() or not on_dir.exists():
        raise FileNotFoundError(
            "Expected folders:\n"
            f"  {off_dir}\n"
            f"  {on_dir}\n"
            "Make sure your release folder structure matches the README."
        )

    print("Downloading GoEmotions (simplified) from HuggingFace...")
    lookup = build_goemotions_lookup()
    print(f"Lookup built: {len(lookup)} rows")

    # Write local verification merges
    merge_and_write("OFF", off_dir, lookup, release_dir / "LOCAL_off_merged_with_english.csv")
    merge_and_write("ON",  on_dir,  lookup, release_dir / "LOCAL_on_merged_with_english.csv")

    print("\nDone. These LOCAL_*.csv files are for verification only and should NOT be uploaded to Zenodo.")


if __name__ == "__main__":
    main()
