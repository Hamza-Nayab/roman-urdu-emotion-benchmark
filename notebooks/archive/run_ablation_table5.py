# ARCHIVED FOR PROVENANCE ONLY. Superseded by scripts/06_run_size_controlled_ablation.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to run the size-controlled ablation experiment (Table 5)
comparing variant-expansion volume gains against spelling robustness.
Runs XLM-R (ON mode, 3 epochs) for k=0, k=3 (full), and k=3 (size-controlled).
Writes results incrementally as they finish.
"""
import re
import json
import time
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# -------------------------
# Path Auto-Detection
# -------------------------
LOCAL_SOURCE = Path("/Users/hamzanayab/Desktop/Assignements/FYP/ToShow./ROman complete/goemotions_roman_urdu_only/goemotions_roman_ur_firstlabel.csv")
KAGGLE_SOURCE = Path("/kaggle/input/research-roman/goemotions_roman_urdu_only/goemotions_roman_ur_firstlabel.csv")

if LOCAL_SOURCE.exists():
    print("Detected LOCAL environment.")
    SOURCE_CSV = LOCAL_SOURCE
    SPLIT_DIR = Path("/Users/hamzanayab/Desktop/Research/Dataset/zenodo_release/data/ablation_on")
    WORK_DIR = Path("/Users/hamzanayab/Desktop/Research/OUTPUTS/ablation_table5")
else:
    print("Detected KAGGLE environment.")
    SOURCE_CSV = KAGGLE_SOURCE
    SPLIT_DIR = Path("/kaggle/input/research-roman/processed_ablation/on")
    WORK_DIR = Path("/kaggle/working/ablation_table5")

MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 128
SEED = 42
LR = 2e-5
TRAIN_BS = 16
EVAL_BS = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
EPOCHS = 3

WORK_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = WORK_DIR / "ablation_table5_results.csv"

# -------------------------
# Helpers
# -------------------------
_re_space = re.compile(r"\s+")
_re_punct = re.compile(r"([!?.,])\1+")

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    t = _re_space.sub(" ", t)
    t = _re_punct.sub(r"\1\1", t)
    t = "".join(ch for ch in t if ch.isprintable())
    return t.strip()

def parse_variants(variants_str: str):
    if not isinstance(variants_str, str) or not variants_str.strip():
        return []
    return [v.strip() for v in variants_str.split("||") if v.strip()]

def load_split_ids(split_dir: Path):
    train_path = split_dir / "train.csv"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"
    
    train_ids = set(pd.read_csv(train_path)["global_id"].astype(str))
    val_ids = set(pd.read_csv(val_path)["global_id"].astype(str))
    test_ids = set(pd.read_csv(test_path)["global_id"].astype(str))
    return train_ids, val_ids, test_ids

def build_view(df_source: pd.DataFrame, ids: set, k: int):
    df = df_source[df_source["global_id"].astype(str).isin(ids)].copy()
    rows = []
    for _, r in df.iterrows():
        gid = str(r["global_id"])
        label = str(r["label_str"]).strip().lower()
        primary = r.get("text_roman_ur", "")
        if isinstance(primary, str) and primary.strip():
            rows.append({"global_id": gid, "text": primary, "label": label})
        if k > 0:
            vars_list = parse_variants(r.get("text_roman_ur_variants", ""))
            for v in parse_variants(r.get("text_roman_ur_variants", ""))[:k]:
                rows.append({"global_id": gid, "text": v, "label": label})
                
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["text"] = out["text"].apply(normalize)
    out = out[out["text"].str.len() > 0]
    out["n_tokens"] = out["text"].str.split().apply(len)
    out = out[(out["n_tokens"] >= 2) & (out["n_tokens"] <= 80)]
    out = out.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    out.drop(columns=["n_tokens"], inplace=True, errors="ignore")
    return out

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, max_length=128):
        self.texts = df["text"].tolist()
        self.labels = [label2id[x] for x in df["label"].tolist()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_length)
        enc["labels"] = self.labels[idx]
        return enc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }

def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    eval_value = kwargs.pop("__eval_strategy_value", "epoch")
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_value
    return TrainingArguments(**kwargs)

# -------------------------
# Main Execution Flow
# -------------------------
def run_experiment():
    print(f"Loading source CSV: {SOURCE_CSV}")
    df_source = pd.read_csv(SOURCE_CSV)
    df_source["global_id"] = df_source["global_id"].astype(str)
    df_source["label_str"] = df_source["label_str"].astype(str).str.strip().str.lower()
    
    train_ids, val_ids, test_ids = load_split_ids(SPLIT_DIR)
    
    # Remove existing summary file to start fresh
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()
        
    configs = [
        {"name": "k=0", "k": 0, "downsample": False},
        {"name": "k=3 (full)", "k": 3, "downsample": False},
        {"name": "k=3 (size-controlled)", "k": 3, "downsample": True},
    ]
    
    for config in configs:
        name = config["name"]
        k = config["k"]
        downsample = config["downsample"]
        
        print(f"\n=======================================================")
        print(f"RUNNING: {name} (k={k}, downsample={downsample})")
        print(f"=======================================================")
        
        set_seed(SEED)
        
        # Build views
        train_df = build_view(df_source, train_ids, k=k)
        val_df = build_view(df_source, val_ids, k=k)
        test_df = build_view(df_source, test_ids, k=k)
        
        if downsample:
            # Downsample train set to match k=0 train set size exactly
            k0_size = len(build_view(df_source, train_ids, k=0))
            train_df = train_df.sample(n=k0_size, random_state=SEED).reset_index(drop=True)
            
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
        
        run_id = f"ablation_table5_{config['name'].replace(' ', '_').replace('=', '_')}"
        args = make_training_args(
            output_dir=str(WORK_DIR / "tmp" / run_id),
            __eval_strategy_value="epoch",
            save_strategy="no",
            learning_rate=LR,
            per_device_train_batch_size=TRAIN_BS,
            per_device_eval_batch_size=EVAL_BS,
            num_train_epochs=EPOCHS,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            logging_steps=300,
            seed=SEED,
            report_to="none",
            fp16=torch.cuda.is_available()
        )
        
        trainer = Trainer(
            model=model,
            args=args,
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
        report = classification_report(y_true, y_pred, target_names=[id2label[i] for i in range(num_labels)], digits=4, zero_division=0)
        
        run_out_dir = WORK_DIR / run_id
        run_out_dir.mkdir(parents=True, exist_ok=True)
        (run_out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
        
        res = {
            "Condition": name,
            "Train Size": len(train_df),
            "Test Macro-F1": test_metrics.get("eval_f1_macro"),
            "Test Accuracy": test_metrics.get("eval_accuracy"),
            "Test Weighted-F1": test_metrics.get("eval_f1_weighted"),
            "Val Macro-F1": val_metrics.get("eval_f1_macro"),
        }
        
        df_step = pd.DataFrame([res])
        write_header = not SUMMARY_CSV.exists()
        df_step.to_csv(SUMMARY_CSV, mode="a" if not write_header else "w", header=write_header, index=False)
        print(f"Saved run results to: {SUMMARY_CSV}")
        
        del trainer, model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("\n=======================================================")
    print("ALL EXPERIMENTS COMPLETED.")
    print(f"Summary table saved to: {SUMMARY_CSV}")

if __name__ == "__main__":
    run_experiment()
