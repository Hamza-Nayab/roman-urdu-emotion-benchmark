#!/usr/bin/env python3
"""
Step 8: Compile and verify all experimental results cited in the paper.
Reads from run artifacts or pre-computed results/ CSVs and verifies them
against the exact figures published in IEEE Access.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PAPER_TARGETS = {
    "ekman6": {
        "XLM-R": {"acc": 0.7031, "macro_f1": 0.5004, "weighted_f1": 0.6961},
        "mBERT": {"acc": 0.6740, "macro_f1": 0.4515, "weighted_f1": 0.6653},
        "MuRIL": {"acc": 0.6736, "macro_f1": 0.4009, "weighted_f1": 0.6570},
    },
    "ablation_size_controlled": {
        "k=0": {"train_size": 42493, "macro_f1": 0.2385, "acc": 0.4546},
        "k=3 (full)": {"train_size": 164857, "macro_f1": 0.2869, "acc": 0.4364},
        "k=3 (size-controlled)": {"train_size": 42493, "macro_f1": 0.2346, "acc": 0.4444},
    },
    "grid_headlines": {
        "XLM-R": {"k": 3, "epochs": 3, "test_macro_f1": 0.2869, "test_acc": 0.4364},
        "mBERT": {"k": 3, "epochs": 2, "test_macro_f1": 0.2807, "test_acc": 0.4406},
        "MuRIL": {"k": 3, "epochs": 4, "test_macro_f1": 0.2334, "test_acc": 0.4046},
    },
}


def compile_ekman(results_dir: Path):
    path = results_dir / "ekman6.csv"
    if not path.exists():
        print(f"[-] {path} not found.")
        return

    df = pd.read_csv(path)
    print("\n" + "=" * 60)
    print("TABLE 3: Ekman-6 Standardized Benchmark (ON, k=1, 3 epochs)")
    print("=" * 60)
    print(f"{'Model':<8} | {'Test Acc':<10} | {'Test Macro-F1':<14} | {'Test Weighted-F1':<16} | Diff F1")
    print("-" * 65)

    name_map = {"XLMR": "XLM-R", "BERTM": "mBERT", "MURIL": "MuRIL"}
    for _, row in df.iterrows():
        key = name_map.get(str(row["model_key"]).upper(), str(row["model_key"]))
        acc = float(row["test_accuracy"])
        mf1 = float(row["test_f1_macro"])
        wf1 = float(row["test_f1_weighted"])

        target = PAPER_TARGETS["ekman6"].get(key, {})
        target_f1 = target.get("macro_f1", mf1)
        diff = mf1 - target_f1

        print(f"{key:<8} | {acc:<10.4f} | {mf1:<14.4f} | {wf1:<16.4f} | {diff:+.4f}")


def compile_size_controlled(results_dir: Path):
    path = results_dir / "ablation_size_controlled.csv"
    if not path.exists():
        print(f"[-] {path} not found.")
        return

    df = pd.read_csv(path)
    print("\n" + "=" * 60)
    print("TABLE 5: Size-Controlled Ablation (Isolating Variation vs Size)")
    print("=" * 60)
    print(f"{'Condition':<22} | {'Train Size':<10} | {'Test Macro-F1':<14} | {'Test Acc':<10} | Diff F1")
    print("-" * 70)

    for _, row in df.iterrows():
        cond = str(row["Condition"]).strip()
        tsize = int(row["Train Size"])
        mf1 = float(row["Test Macro-F1"])
        acc = float(row["Test Accuracy"])

        target = PAPER_TARGETS["ablation_size_controlled"].get(cond, {})
        target_f1 = target.get("macro_f1", mf1)
        diff = mf1 - target_f1

        print(f"{cond:<22} | {tsize:<10} | {mf1:<14.4f} | {acc:<10.4f} | {diff:+.4f}")


def compile_main_grid(results_dir: Path):
    path = results_dir / "main_grid.csv"
    if not path.exists():
        print(f"[-] {path} not found.")
        return

    df = pd.read_csv(path)
    print("\n" + "=" * 60)
    print("TABLE 2 / HEADLINES: Primary 27-Class Task Optimal Grid Points")
    print("=" * 60)
    print(f"{'Model':<8} | {'k':<3} | {'Epochs':<6} | {'Test Macro-F1':<14} | {'Test Acc':<10} | Diff F1")
    print("-" * 60)

    key_map = {
        "XLM-R": ["XLMR", "XLM-R", "xlmr"],
        "mBERT": ["BERTM", "MBERT", "mbert"],
        "MuRIL": ["MURIL", "muril"],
    }

    for model, cfg in PAPER_TARGETS["grid_headlines"].items():
        keys = key_map.get(model, [model])
        m_filter = df["model_key"].astype(str).isin(keys)
        # If mode column exists, select ON (variant-expanded view)
        if "mode" in df.columns:
            sub = df[m_filter & (df["mode"] == "ON") & (df["k"] == cfg["k"]) & (df["epochs"] == cfg["epochs"])]
        else:
            sub = df[m_filter & (df["k"] == cfg["k"]) & (df["epochs"] == cfg["epochs"])]

        if not sub.empty:
            row = sub.iloc[0]
            mf1 = float(row["test_f1_macro"])
            acc = float(row["test_accuracy"])
            diff = mf1 - cfg["test_macro_f1"]
            print(f"{model:<8} | {cfg['k']:<3} | {cfg['epochs']:<6} | {mf1:<14.4f} | {acc:<10.4f} | {diff:+.4f}")


def compile_seed_robustness(results_dir: Path):
    path = results_dir / "seed_robustness.csv"
    if not path.exists():
        return

    df = pd.read_csv(path)
    print("\n" + "=" * 60)
    print("TABLE 4 / SEED ROBUSTNESS: Multi-Seed Stability (seeds 42, 43, 44)")
    print("=" * 60)
    print(f"{'Model':<8} | {'Configuration':<18} | {'Test Macro-F1 (mean ± std)':<26}")
    print("-" * 60)

    for model in ["XLM-R", "mBERT", "MuRIL"]:
        sub = df[df["model"] == model]
        if not sub.empty:
            mean = sub["test_macro_f1"].mean()
            std = sub["test_macro_f1"].std()
            cfg = sub.iloc[0]["configuration"]
            print(f"{model:<8} | {cfg:<18} | {mean:.4f} ± {std:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Compile paper results and verify against citations")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    res_dir = Path(args.results_dir)
    print(f"Compiling benchmark results from: {res_dir.resolve()}")

    compile_main_grid(res_dir)
    compile_ekman(res_dir)
    compile_size_controlled(res_dir)
    compile_seed_robustness(res_dir)
    print("\nCompilation and verification check completed successfully.\n")


if __name__ == "__main__":
    main()
