#!/usr/bin/env python3
"""
Step 6: Size-controlled ablation experiment (Replicating Table 5).
Compares variant-expansion volume gains against spelling robustness.
Runs XLM-R (3 epochs) across 3 conditions:
  1. k=0 (primary only, train size: 42,493)
  2. k=3 (full variant expansion, train size: 164,857)
  3. k=3 (size-controlled, downsampled to match k=0 size: 42,493)
Writes results incrementally to ablation_table5_results.csv.
"""

import os
import json
import time
import inspect
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from romanurdu.views import build_view
from romanurdu.splits import load_split_ids
from romanurdu.evaluate import compute_metrics
from romanurdu.train import TextDataset, make_training_args

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
LR = 2e-5
TRAIN_BS = 16
EVAL_BS = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
EPOCHS = 3


def main():
    parser = argparse.ArgumentParser(description="Size-controlled ablation (Table 5)")
    parser.add_argument("--source-csv", type=str, default="data/goemotions_roman_ur_firstlabel.csv")
    parser.add_argument("--split-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="outputs/ablation_size_controlled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "ablation_table5_results.csv"

    source_path = Path(args.source_csv)
    split_path = Path(args.split_dir)

    print(f"Loading source CSV: {source_path}")
    df_source = pd.read_csv(source_path)
    df_source["global_id"] = df_source["global_id"].astype(str)
    label_col = "label_str" if "label_str" in df_source.columns else "label"
    df_source[label_col] = df_source[label_col].astype(str).str.strip().str.lower()

    train_ids, val_ids, test_ids = load_split_ids(split_path)

    configs = [
        {"name": "k=0", "k": 0, "downsample": False},
        {"name": "k=3 (full)", "k": 3, "downsample": False},
        {"name": "k=3 (size-controlled)", "k": 3, "downsample": True},
    ]

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        DataCollatorWithPadding,
        set_seed,
    )
    from sklearn.metrics import classification_report

    for config in configs:
        name = config["name"]
        k = config["k"]
        downsample = config["downsample"]
        run_id = f"ablation_table5_{name.replace(' ', '_').replace('=', '_').replace('(', '').replace(')', '')}"
        run_out_dir = out_dir / run_id
        run_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=======================================================")
        print(f"RUNNING: {name} (k={k}, downsample={downsample})")
        print(f"=======================================================")

        set_seed(args.seed)

        # Build views with primary task length filtering [2, 80]
        train_df = build_view(df_source, train_ids, k=k, apply_length_filter=True)
        val_df = build_view(df_source, val_ids, k=k, apply_length_filter=True)
        test_df = build_view(df_source, test_ids, k=k, apply_length_filter=True)

        if downsample:
            # Downsample train set to match k=0 train set size exactly (42,493)
            k0_size = len(build_view(df_source, train_ids, k=0, apply_length_filter=True))
            train_df = train_df.sample(n=k0_size, random_state=args.seed).reset_index(drop=True)

        print(f"Dataset sizes -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

        labels = sorted(train_df["label"].unique().tolist())
        label2id = {lab: i for i, lab in enumerate(labels)}
        id2label = {i: lab for lab, i in label2id.items()}
        num_labels = len(label2id)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
        )

        train_ds = TextDataset(train_df, tokenizer, label2id, max_length=MAX_LENGTH)
        val_ds = TextDataset(val_df, tokenizer, label2id, max_length=MAX_LENGTH)
        test_ds = TextDataset(test_df, tokenizer, label2id, max_length=MAX_LENGTH)

        train_args = make_training_args(
            output_dir=str(out_dir / "tmp" / run_id),
            __eval_strategy_value="epoch",
            save_strategy="no",
            learning_rate=LR,
            per_device_train_batch_size=TRAIN_BS,
            per_device_eval_batch_size=EVAL_BS,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            logging_steps=300,
            seed=args.seed,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        val_metrics = trainer.evaluate(val_ds)
        test_metrics = trainer.evaluate(test_ds)

        preds = trainer.predict(test_ds)
        y_true = preds.label_ids
        y_pred = np.argmax(preds.predictions, axis=1)
        report = classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in range(num_labels)],
            digits=4,
            zero_division=0,
        )
        (run_out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

        res = {
            "Condition": name,
            "Train Size": len(train_df),
            "Test Macro-F1": test_metrics.get("eval_f1_macro"),
            "Test Accuracy": test_metrics.get("eval_accuracy"),
            "Test Weighted-F1": test_metrics.get("eval_f1_weighted"),
            "Val Macro-F1": val_metrics.get("eval_f1_macro"),
        }

        # Append run results incrementally to summary CSV
        df_step = pd.DataFrame([res])
        write_header = not summary_csv.exists()
        df_step.to_csv(summary_csv, mode="a" if not write_header else "w", header=write_header, index=False)
        print(f"Saved run results to: {summary_csv}")

        del trainer, model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=======================================================")
    print("ALL ABLATION EXPERIMENTS COMPLETED.")
    print(f"Summary table saved to: {summary_csv}")


if __name__ == "__main__":
    main()
