"""
Majority-class and random-uniform baseline calculations for emotion classification.

Provides both closed-form and empirical calculations for the 28-class / 27-class
GoEmotions benchmark and the 6-class Ekman benchmark.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def compute_majority_baseline(
    train_labels: List[Any],
    test_labels: List[Any],
    num_classes: int = None,
) -> Dict[str, float]:
    """
    Compute majority class prediction floor (model always predicts train modal class).

    Args:
        train_labels: List of training labels.
        test_labels: List of test labels.
        num_classes: Total number of classes (if known/fixed). If None, inferred from union.

    Returns:
        Dictionary containing 'accuracy', 'f1_macro', 'f1_weighted', 'majority_label'.
    """
    train_series = pd.Series(train_labels)
    test_series = pd.Series(test_labels)

    majority_label = train_series.mode()[0]
    preds = [majority_label] * len(test_series)

    all_labels = sorted(list(set(train_series.unique()) | set(test_series.unique())))
    if num_classes is not None and len(all_labels) < num_classes:
        labels_arg = list(range(num_classes)) if isinstance(all_labels[0], int) else all_labels
    else:
        labels_arg = all_labels

    acc = accuracy_score(test_series, preds)
    f1_macro = f1_score(test_series, preds, labels=labels_arg, average="macro", zero_division=0)
    f1_weighted = f1_score(test_series, preds, labels=labels_arg, average="weighted", zero_division=0)

    return {
        "majority_label": str(majority_label),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }


def compute_uniform_baseline(
    test_labels: List[Any],
    num_classes: int,
    seed: int = 42,
    num_simulations: int = 100,
) -> Dict[str, float]:
    """
    Compute random uniform baseline (expectation: accuracy = 1/C, macro-F1 = 1/C).

    Args:
        test_labels: List of test labels.
        num_classes: Total number of classes C.
        seed: Random seed for Monte Carlo simulation.
        num_simulations: Number of Monte Carlo runs to average.

    Returns:
        Dictionary containing empirical and closed-form accuracy, f1_macro, f1_weighted.
    """
    rng = np.random.RandomState(seed)
    unique_labels = sorted(list(set(test_labels)))

    # Closed-form theoretical floors
    theoretical_floor = 1.0 / num_classes

    # Monte Carlo empirical estimate
    accs, macro_f1s, weighted_f1s = [], [], []
    test_arr = np.array(test_labels)
    n = len(test_arr)

    for _ in range(num_simulations):
        preds = rng.choice(unique_labels, size=n)
        accs.append(accuracy_score(test_arr, preds))
        macro_f1s.append(f1_score(test_arr, preds, average="macro", zero_division=0))
        weighted_f1s.append(f1_score(test_arr, preds, average="weighted", zero_division=0))

    return {
        "theoretical_accuracy": theoretical_floor,
        "theoretical_f1_macro": theoretical_floor,
        "simulated_accuracy": float(np.mean(accs)),
        "simulated_f1_macro": float(np.mean(macro_f1s)),
        "simulated_f1_weighted": float(np.mean(weighted_f1s)),
    }
