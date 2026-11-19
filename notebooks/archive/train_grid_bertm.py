# ARCHIVED FOR PROVENANCE ONLY. Superseded by scripts/04_run_grid.py and src/romanurdu/train.py
# ============================================================
# Q1-STANDALONE (NEW NOTEBOOK): BERT-Multilingual (bert-base-multilingual-cased)
# OFF + ON splits + k={0,1,3} + epochs={2,3,4}
#
# Outputs (same structure as your XLM-R runs):
# 1) Per-run artifacts:
#    /kaggle/working/grid_runs_artifacts/{MODE}/mode_{MODE}_k{k}_e{epochs}/
#      - val_metrics.json
#      - test_metrics.json
#      - classification_report.txt
#      - confusion_matrix.csv
# 2) Single JSONL log (appends; DOES NOT reset):
#    /kaggle/working/RUN_LOG.jsonl
# 3) One best model kept:
#    /kaggle/working/best_bertm_model/ + best_run.json
# 4) Summary CSV:
#    /kaggle/working/q1_bertm_summary.csv
# ============================================================
import re
import json
import time
import inspect
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
    TrainerCallback,
)

BASE_DIR = Path("/kaggle/input/research-roman")
SOURCE_CSV = BASE_DIR / "goemotions_roman_urdu_only" / "goemotions_roman_ur_firstlabel.csv"
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128
SEED = 42
LR = 2e-5
TRAIN_BS = 16
EVAL_BS = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
K_LIST = [0, 1, 3]
EPOCH_LIST = [2, 3, 4]
