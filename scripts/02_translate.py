#!/usr/bin/env python3
"""
Step 2: English -> Roman Urdu translation pipeline with Gemma 3 (via Ollama).
Performs chunked incremental translation, deterministic spelling variant expansion,
and QA anomaly flagging.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from romanurdu.translate import (
    GemmaRomanUrduTranslator,
    convert_to_roman_urdu_with_variants,
)


def load_progress(output_csv: Path) -> int:
    if not output_csv.exists():
        return 0
    return len(pd.read_csv(output_csv))


def main():
    parser = argparse.ArgumentParser(description="Incremental Roman Urdu translation pipeline")
    parser.add_argument("--source-csv", type=str, default="data/goemotions_en_all_firstlabel.csv")
    parser.add_argument("--output-csv", type=str, default="data/goemotions_roman_ur_firstlabel.csv")
    parser.add_argument("--flagged-csv", type=str, default="data/goemotions_roman_ur_flagged.csv")
    parser.add_argument("--ollama-model", type=str, default="gemma3:latest")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--max-variants", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    source_path = Path(args.source_csv)
    output_path = Path(args.output_csv)
    flagged_path = Path(args.flagged_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    flagged_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        logging.error("Source CSV missing: %s. Run scripts/01_download_goemotions.py first.", source_path)
        sys.exit(1)

    source_df = pd.read_csv(source_path)
    total_rows = len(source_df)
    translator = GemmaRomanUrduTranslator(model_name=args.ollama_model, timeout=args.timeout)

    print(f"Translating {total_rows} rows from {source_path}...")

    while True:
        start_idx = load_progress(output_path)
        if start_idx >= total_rows:
            print("All rows successfully translated.")
            break

        end_idx = min(start_idx + args.chunk_size, total_rows)
        batch_df = source_df.iloc[start_idx:end_idx].copy()

        roman_records = []
        flagged_records = []

        for _, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Chunk {start_idx}-{end_idx}", unit="row"):
            try:
                text_en = str(row["text_en"])
                primary, variants, reason = convert_to_roman_urdu_with_variants(
                    text_en=text_en,
                    translator=translator,
                    max_variants=args.max_variants,
                )
                variants_str = "||".join(variants) if variants else ""

                record = {
                    "global_id": row["global_id"],
                    "split": row["split"],
                    "idx": row["idx"],
                    "comment_id": row["comment_id"],
                    "text_en": text_en,
                    "labels_ids": row["labels_ids"],
                    "labels_str": row["labels_str"],
                    "label": row["label"],
                    "label_str": row["label_str"],
                    "text_roman_ur": primary,
                    "text_roman_ur_variants": variants_str,
                }
                roman_records.append(record)

                if reason:
                    flagged = dict(record)
                    flagged["flag_reason"] = reason
                    flagged_records.append(flagged)

            except Exception as e:
                logging.error("Failed processing row %s: %s", row["global_id"], e)
                continue

        # Append batch
        write_header = not output_path.exists()
        pd.DataFrame(roman_records).to_csv(
            output_path,
            mode="a" if not write_header else "w",
            header=write_header,
            index=False,
        )

        if flagged_records:
            flagged_header = not flagged_path.exists()
            pd.DataFrame(flagged_records).to_csv(
                flagged_path,
                mode="a" if not flagged_header else "w",
                header=flagged_header,
                index=False,
            )

        print(f"Processed chunk {end_idx}/{total_rows}")


if __name__ == "__main__":
    main()
