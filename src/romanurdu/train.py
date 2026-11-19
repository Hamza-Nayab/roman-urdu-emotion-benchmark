"""
Unified training entry point for Roman Urdu emotion classification.

Supports:
- Backbones: xlmr (xlm-roberta-base), mbert (bert-base-multilingual-cased), muril (google/muril-base-cased)
- Tasks: primary (28-class GoEmotions), ekman6 (6-class Ekman mapped)
- Variant admission: k in {0, 1, 3}
- Idempotent execution: skips completed runs when all 4 artifacts exist
- Per-epoch JSONL logging and reproducibility artifact persistence
"""

import os
import re
import json
import time
import inspect
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Set

import numpy as np
import pandas as pd

from .views import build_view
from .splits import load_split_ids
from .ekman import to_ekman, EKMAN_LABELS, EKMAN_LABEL2ID, EKMAN_ID2LABEL
from .evaluate import compute_metrics, save_evaluation_artifacts

# Prevent OpenMP / thread deadlock on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MODEL_REGISTRY: Dict[str, str] = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "muril": "google/muril-base-cased",
}


def make_training_args(**kwargs):
    """
    Construct TrainingArguments with backward/forward compatibility across
    HuggingFace Transformers versions (eval_strategy vs evaluation_strategy).
    """
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    eval_value = kwargs.pop("__eval_strategy_value", "epoch")

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_value
    else:
        raise RuntimeError("Neither eval_strategy nor evaluation_strategy supported by installed Transformers.")

    return TrainingArguments(**kwargs)


def is_run_completed(run_dir: Path) -> bool:
    """
    Idempotency check: returns True if all 4 required artifacts exist.
    """
    run_dir = Path(run_dir)
    return (
        (run_dir / "val_metrics.json").exists()
        and (run_dir / "test_metrics.json").exists()
        and (run_dir / "classification_report.txt").exists()
        and (run_dir / "confusion_matrix.csv").exists()
    )


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Safely append JSON record to JSONL log file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_epoch_logger_class():
    from transformers import TrainerCallback

    class SingleFileEpochLogger(TrainerCallback):
        """Callback logging validation metrics at the end of each evaluation epoch."""

        def __init__(self, logfile: Path, run_id: str, model_key: str, task: str, k: int, total_epochs: int):
            self.logfile = logfile
            self.run_id = run_id
            self.model_key = model_key
            self.task = task
            self.k = k
            self.total_epochs = total_epochs

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None or state.epoch is None:
                return
            append_jsonl(
                self.logfile,
                {
                    "type": "epoch_eval",
                    "timestamp": time.time(),
                    "run_id": self.run_id,
                    "model_key": self.model_key,
                    "task": self.task,
                    "k": self.k,
                    "epoch": int(state.epoch),
                    "total_epochs": int(self.total_epochs),
                    "eval_loss": metrics.get("eval_loss"),
                    "eval_accuracy": metrics.get("eval_accuracy"),
                    "eval_f1_macro": metrics.get("eval_f1_macro"),
                    "eval_f1_weighted": metrics.get("eval_f1_weighted"),
                },
            )

    return SingleFileEpochLogger


class TextDataset:
    """Sequence classification dataset."""

    def __init__(self, df: pd.DataFrame, tokenizer, label2id: Dict[str, int], max_length: int = 128):
        self.texts = df["text"].tolist()
        self.labels = [int(label2id[str(x)]) for x in df["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)
        enc["labels"] = self.labels[idx]
        return enc


def run_training(
    source_csv: Path,
    split_dir: Path,
    output_dir: Path,
    model_key: str = "xlmr",
    task: str = "primary",
    k: int = 0,
    epochs: int = 3,
    seed: int = 42,
    lr: float = 2e-5,
    train_bs: int = 16,
    eval_bs: int = 32,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    max_length: int = 128,
    log_file: Optional[Path] = None,
    force_rerun: bool = False,
    apply_length_filter: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Execute one complete training and evaluation run.
    """
    source_csv = Path(source_csv)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        DataCollatorWithPadding,
        set_seed,
        EarlyStoppingCallback,
    )
    SingleFileEpochLogger = get_epoch_logger_class()

    set_seed(seed)

    model_key = model_key.lower()
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_key '{model_key}'. Supported: {list(MODEL_REGISTRY.keys())}")
    hf_model_name = MODEL_REGISTRY[model_key]

    # Task-specific configurations
    if task == "primary":
        run_name = f"{model_key}_k{k}_e{epochs}_seed{seed}"
        run_dir = output_dir / run_name
        if apply_length_filter is None:
            apply_length_filter = True
    elif task == "ekman6":
        run_name = f"ekman6_{model_key}_k{k}_e{epochs}_seed{seed}"
        run_dir = output_dir / run_name
        if apply_length_filter is None:
            apply_length_filter = False
    else:
        raise ValueError(f"Unknown task '{task}'. Supported: 'primary', 'ekman6'")

    # Idempotent skip check
    if not force_rerun and is_run_completed(run_dir):
        print(f"SKIP (run already completed): {run_dir}")
        with open(run_dir / "test_metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)

    if log_file is None:
        log_file = output_dir / "RUN_LOG.jsonl"

    # Load source data and split IDs
    df_source = pd.read_csv(source_csv)
    train_ids, val_ids, test_ids = load_split_ids(split_dir)

    # Build views
    train_df = build_view(df_source, train_ids, k=k, apply_length_filter=apply_length_filter)
    val_df = build_view(df_source, val_ids, k=k, apply_length_filter=apply_length_filter)
    test_df = build_view(df_source, test_ids, k=k, apply_length_filter=apply_length_filter)

    # Apply Ekman mapping if task is ekman6
    dropped_neutral = {}
    if task == "ekman6":
        train_df, b1, a1 = to_ekman(train_df)
        val_df, b2, a2 = to_ekman(val_df)
        test_df, b3, a3 = to_ekman(test_df)
        dropped_neutral = {"train": b1 - a1, "val": b2 - a2, "test": b3 - a3}
        labels = EKMAN_LABELS
        label2id = EKMAN_LABEL2ID
        id2label = EKMAN_ID2LABEL
    else:
        labels = sorted(train_df["label"].unique().tolist())
        label2id = {lab: i for i, lab in enumerate(labels)}
        id2label = {i: lab for lab, i in label2id.items()}

    num_labels = len(label2id)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    train_ds = TextDataset(train_df, tokenizer, label2id, max_length=max_length)
    val_ds = TextDataset(val_df, tokenizer, label2id, max_length=max_length)
    test_ds = TextDataset(test_df, tokenizer, label2id, max_length=max_length)

    tmp_ckpt_dir = run_dir / "checkpoints"
    save_strategy = "epoch" if task == "ekman6" else "no"

    kwargs = {
        "output_dir": str(tmp_ckpt_dir),
        "__eval_strategy_value": "epoch",
        "save_strategy": save_strategy,
        "learning_rate": lr,
        "per_device_train_batch_size": train_bs,
        "per_device_eval_batch_size": eval_bs,
        "num_train_epochs": epochs,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "logging_steps": 200,
        "seed": seed,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
    }

    callbacks = [SingleFileEpochLogger(log_file, run_name, model_key, task, k, epochs)]
    if task == "ekman6":
        kwargs["save_total_limit"] = 1
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "eval_f1_macro"
        kwargs["greater_is_better"] = True
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=1))

    args = make_training_args(**kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    t0 = time.time()
    append_jsonl(
        log_file,
        {
            "type": "run_start",
            "timestamp": t0,
            "run_id": run_name,
            "model_key": model_key,
            "task": task,
            "k": k,
            "epochs": epochs,
            "seed": seed,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "dropped_neutral": dropped_neutral,
        },
    )

    trainer.train()
    elapsed = time.time() - t0

    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)

    preds = trainer.predict(test_ds)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=1)

    target_names = [id2label[i] for i in range(num_labels)]
    save_evaluation_artifacts(
        y_true=y_true,
        y_pred=y_pred,
        target_names=target_names,
        output_dir=run_dir,
        metrics=test_metrics,
        prefix="test",
    )

    with open(run_dir / "val_metrics.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    val_f1 = float(val_metrics.get("eval_f1_macro", -1.0))
    test_f1 = float(test_metrics.get("eval_f1_macro", -1.0))

    summary = {
        "run_id": run_name,
        "model_key": model_key,
        "task": task,
        "k": int(k),
        "epochs": int(epochs),
        "seed": int(seed),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "val_f1_macro": val_f1,
        "val_accuracy": float(val_metrics.get("eval_accuracy", -1.0)),
        "val_f1_weighted": float(val_metrics.get("eval_f1_weighted", -1.0)),
        "test_f1_macro": test_f1,
        "test_accuracy": float(test_metrics.get("eval_accuracy", -1.0)),
        "test_f1_weighted": float(test_metrics.get("eval_f1_weighted", -1.0)),
        "elapsed_sec": float(elapsed),
        "run_dir": str(run_dir),
    }

    append_jsonl(
        log_file,
        {
            "type": "run_end",
            "timestamp": time.time(),
            "summary": summary,
        },
    )

    print(f"COMPLETED: {run_dir}")
    print(f"VAL F1-macro: {val_f1:.4f} | TEST F1-macro: {test_f1:.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Unified Trainer for Roman Urdu Emotion Classification")
    parser.add_argument("--model", type=str, default="xlmr", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--task", type=str, default="primary", choices=["primary", "ekman6"])
    parser.add_argument("--k", type=int, default=0, choices=[0, 1, 3])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-bs", type=int, default=16)
    parser.add_argument("--eval-bs", type=int, default=32)
    parser.add_argument("--source-csv", type=str, default="data/goemotions_roman_ur_firstlabel.csv")
    parser.add_argument("--split-dir", type=str, default="data/splits")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    run_training(
        source_csv=Path(args.source_csv),
        split_dir=Path(args.split_dir),
        output_dir=Path(args.output_dir),
        model_key=args.model,
        task=args.task,
        k=args.k,
        epochs=args.epochs,
        seed=args.seed,
        lr=args.lr,
        train_bs=args.train_bs,
        eval_bs=args.eval_bs,
        force_rerun=args.force_rerun,
    )


if __name__ == "__main__":
    main()
