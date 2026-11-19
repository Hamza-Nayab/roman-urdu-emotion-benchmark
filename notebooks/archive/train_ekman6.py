# ARCHIVED FOR PROVENANCE ONLY. Superseded by scripts/05_run_ekman6.py and src/romanurdu/train.py
# ============================================================
# EKMAN-6 DIRECT TRAINING (MAIN BENCHMARK TABLE)
# Models: XLM-R, MuRIL, mBERT
# Setting: ON split only, k=1 variants, epochs=3 (standardized)
# ============================================================
import os, re, json, time, inspect
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
    EarlyStoppingCallback,
)

BASE_DIR = Path("/kaggle/input/research-roman")
SOURCE_CSV = BASE_DIR / "goemotions_roman_urdu_only" / "goemotions_roman_ur_firstlabel.csv"
SPLIT_DIR = BASE_DIR / "processed_ablation" / "on"
OUT_DIR = Path("/kaggle/working/ekman6_direct")
SEED = 42
K_VARIANTS = 1
EPOCHS = 3
MAX_LENGTH = 128
LR = 2e-5
TRAIN_BS = 16
EVAL_BS = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MODELS = {
    "XLMR": "xlm-roberta-base",
    "MURIL": "google/muril-base-cased",
    "BERTM": "bert-base-multilingual-cased",
}
