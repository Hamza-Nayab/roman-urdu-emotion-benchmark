#!/usr/bin/env python3
"""
Step 1: Download the GoEmotions dataset (simplified config) from HuggingFace,
extract primary emotion labels, and generate deterministic global_id assignments.
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from romanurdu.translate import get_primary_label

DATASET_NAME = "go_emotions"
DATASET_CONFIG = "simplified"


def main():
    parser = argparse.ArgumentParser(description="Download and format GoEmotions simplified dataset")
    parser.add_argument("--output-csv", type=str, default="data/goemotions_en_all_firstlabel.csv")
    args = parser.parse_args()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logging.info("Target file already exists: %s", out_path)
        print(f"File already exists: {out_path}")
        return

    print(f"Downloading {DATASET_NAME} ({DATASET_CONFIG}) from HuggingFace...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    go_label_names = dataset["train"].features["labels"].feature.names

    dfs = []
    for split_name in ["train", "validation", "test"]:
        if split_name not in dataset:
            continue
        ds_split = dataset[split_name]
        tmp_df = ds_split.to_pandas()
        tmp_df["split"] = split_name
        tmp_df["labels_ids"] = tmp_df["labels"].apply(lambda ids: ",".join(str(i) for i in ids))
        tmp_df["labels_str"] = tmp_df["labels"].apply(lambda ids: ",".join(go_label_names[i] for i in ids))

        def _map_row(ids):
            lid, lstr = get_primary_label(ids, go_label_names)
            return pd.Series({"label": lid, "label_str": lstr})

        mapped = tmp_df["labels"].apply(_map_row)
        tmp_df = pd.concat([tmp_df, mapped], axis=1)
        tmp_df = tmp_df.dropna(subset=["label"]).reset_index(drop=True)
        tmp_df["idx"] = tmp_df.index
        tmp_df.rename(columns={"id": "comment_id", "text": "text_en"}, inplace=True)
        dfs.append(tmp_df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["global_id"] = df_all["split"] + "_" + df_all["idx"].astype(str).str.zfill(6)
    df_all = df_all[
        ["global_id", "split", "idx", "comment_id", "text_en", "labels_ids", "labels_str", "label", "label_str"]
    ]

    df_all.to_csv(out_path, index=False)
    print(f"Successfully saved {len(df_all)} source rows to {out_path}")


if __name__ == "__main__":
    main()
