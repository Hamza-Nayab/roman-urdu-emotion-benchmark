"""
Evaluation metrics, classification report generation, and confusion matrix export.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Standard HuggingFace Trainer metric callback.
    Computes accuracy, macro-averaged F1, and weighted-averaged F1.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


def save_evaluation_artifacts(
    y_true: List[int],
    y_pred: List[int],
    target_names: List[str],
    output_dir: Path,
    metrics: Dict[str, Any] = None,
    prefix: str = "test",
) -> None:
    """
    Export evaluation artifacts required by the reproducibility protocol:
    1. {prefix}_metrics.json
    2. classification_report.txt
    3. confusion_matrix.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classification report
    rep = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(rep, encoding="utf-8")

    # 2. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_names))))
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(output_dir / "confusion_matrix.csv", index=True)

    # 3. Metrics JSON
    if metrics is not None:
        with open(output_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
