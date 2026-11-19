#!/usr/bin/env python3
"""
Step 4: Execute the 27-configuration benchmark grid across 3 backbones (XLM-R, mBERT, MuRIL),
3 variant levels (k in {0, 1, 3}), and 3 epoch budgets (epochs in {2, 3, 4}).
Fully idempotent: skips completed runs automatically.
"""

import argparse
from pathlib import Path
import pandas as pd
from romanurdu.train import run_training, MODEL_REGISTRY

K_LIST = [0, 1, 3]
EPOCH_LIST = [2, 3, 4]


def main():
    parser = argparse.ArgumentParser(description="Run full benchmark grid across backbones, k, and epochs")
    parser.add_argument("--model", type=str, default="all", choices=["all", "xlmr", "mbert", "muril"])
    parser.add_argument("--source-csv", type=str, default="data/goemotions_roman_ur_firstlabel.csv")
    parser.add_argument("--split-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="outputs/grid")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    models = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "grid_summary.csv"

    rows = []
    print(f"Starting grid execution for models: {models}")
    print(f"k in {K_LIST}, epochs in {EPOCH_LIST}")

    for model_key in models:
        for k in K_LIST:
            for ep in EPOCH_LIST:
                print(f"\n>>> Running configuration: model={model_key}, k={k}, epochs={ep} <<<")
                res = run_training(
                    source_csv=Path(args.source_csv),
                    split_dir=Path(args.split_dir),
                    output_dir=out_dir,
                    model_key=model_key,
                    task="primary",
                    k=k,
                    epochs=ep,
                    seed=42,
                    force_rerun=args.force_rerun,
                    apply_length_filter=True,
                )
                rows.append(res)
                # Incrementally update summary CSV
                pd.DataFrame(rows).to_csv(summary_csv, index=False)

    print(f"\nGrid execution complete. Summary saved to: {summary_csv}")


if __name__ == "__main__":
    main()
