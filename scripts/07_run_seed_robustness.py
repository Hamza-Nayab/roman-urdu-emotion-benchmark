#!/usr/bin/env python3
"""
Step 7: Seed robustness evaluation across 5 random seeds (Replicating Table 6).
Evaluates XLM-R under the optimal configuration (k=3, 3 epochs) across seeds [42, 123, 456, 789, 1011].
Computes mean and standard deviation for Test Macro-F1, Accuracy, and Weighted-F1.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from romanurdu.train import run_training

SEEDS = [42, 43, 44]


def main():
    parser = argparse.ArgumentParser(description="Multi-seed robustness evaluation (Table 6)")
    parser.add_argument("--source-csv", type=str, default="data/goemotions_roman_ur_firstlabel.csv")
    parser.add_argument("--split-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="outputs/seed_robustness")
    parser.add_argument("--model", type=str, default="xlmr")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "seed_robustness_summary.csv"

    rows = []
    print(f"Running seed robustness for {args.model} (k={args.k}, epochs={args.epochs}) across seeds: {SEEDS}")

    for s in SEEDS:
        print(f"\n--- Running Seed {s} ---")
        res = run_training(
            source_csv=Path(args.source_csv),
            split_dir=Path(args.split_dir),
            output_dir=out_dir,
            model_key=args.model,
            task="primary",
            k=args.k,
            epochs=args.epochs,
            seed=s,
            force_rerun=args.force_rerun,
            apply_length_filter=True,
        )
        rows.append(res)
        pd.DataFrame(rows).to_csv(summary_csv, index=False)

    df = pd.DataFrame(rows)
    f1_mean, f1_std = df["test_f1_macro"].mean(), df["test_f1_macro"].std()
    acc_mean, acc_std = df["test_accuracy"].mean(), df["test_accuracy"].std()
    w_mean, w_std = df["test_f1_weighted"].mean(), df["test_f1_weighted"].std()

    print("\n=======================================================")
    print(f"Seed Robustness Results across {len(SEEDS)} seeds:")
    print(f"  Test Macro-F1:    {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  Test Accuracy:    {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Test Weighted-F1: {w_mean:.4f} ± {w_std:.4f}")
    print("=======================================================")
    print(f"Full results saved to: {summary_csv}")


if __name__ == "__main__":
    main()
