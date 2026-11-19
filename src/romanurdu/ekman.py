"""
Ekman-6 emotion mapping and conversion utilities.

Standardizes 27 GoEmotions fine-grained categories into Ekman's 6 basic emotion
families: anger, disgust, fear, joy, sadness, and surprise (neutral is excluded).
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd

GOEMO_TO_EKMAN: Dict[str, Optional[str]] = {
    # Anger
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    # Disgust
    "disgust": "disgust",
    # Fear
    "fear": "fear",
    "nervousness": "fear",
    # Joy
    "joy": "joy",
    "amusement": "joy",
    "approval": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "admiration": "joy",
    "caring": "joy",
    "desire": "joy",
    # Sadness
    "sadness": "sadness",
    "disappointment": "sadness",
    "embarrassment": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    # Surprise
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    # Excluded
    "neutral": None,
}

EKMAN_LABELS: List[str] = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
]

EKMAN_LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(EKMAN_LABELS)}
EKMAN_ID2LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(EKMAN_LABELS)}


def to_ekman(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Project fine-grained emotion labels to the 6 Ekman categories.
    Excludes instances mapped to None (e.g. 'neutral').

    Args:
        df: DataFrame containing a 'label' or 'label_str' column.

    Returns:
        Tuple of (ekman_df, before_count, after_count).
        The returned DataFrame contains ['global_id', 'text', 'label', 'label_id'].
    """
    out = df.copy()
    label_col = "label" if "label" in out.columns else "label_str"

    out["label_raw"] = out[label_col].astype(str).str.strip().str.lower()
    out["ekman"] = out["label_raw"].map(GOEMO_TO_EKMAN)

    before_count = len(out)
    out = out[out["ekman"].notna()].reset_index(drop=True)
    after_count = len(out)

    out["label"] = out["ekman"]
    out["label_id"] = out["label"].map(EKMAN_LABEL2ID).astype(int)

    cols = [c for c in ["global_id", "text", "label", "label_id"] if c in out.columns]
    out = out[cols]

    return out, before_count, after_count
