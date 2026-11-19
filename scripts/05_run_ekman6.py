#!/usr/bin/env python3
"""
Step 5: Execute Ekman-6 standardized benchmark across the 3 backbones.
Standardized setting: k=1 variant, 3 epochs, early stopping patience=1.
Replicates Table 3 of the paper.
"""

import argparse
from pathlib import Path
import pandas as pd
from romanurdu.train import run_training, MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser(description="Run Ekman-6 standardized benchmark")
    parser.add_argument("--source-csv", type=str, default="data/goemotions_roman_ur_firstlabel.csv")
    parser.add_argument("--split-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="outputs/ekman6")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "ekman6_main_table.csv"

    rows = []
    models = ["xlmr", "mbert", "muril"]

    for model_key in models:
        print(f"\n>>> Running Ekman-6: model={model_key}, k=1, epochs=3 <<<")
        res = run_training(
            source_csv=Path(args.source_csv),
            split_dir=Path(args.split_dir),
            output_dir=out_dir,
            model_key=model_key,
            task="ekman6",
            k=1,
            epochs=3,
            seed=42,
            force_rerun=args.force_rerun,
            apply_length_filter=False,
        )
        rows.append(res)
        pd.DataFrame(rows).to_csv(summary_csv, index=False)

    df_res = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False).reset_index(drop=True)
    df_res.to_csv(summary_csv, index=False)
    print(f"\nEkman-6 execution complete. Saved main table to: {summary_csv}")


if __name__ == "__main__":
    main()
