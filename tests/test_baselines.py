"""
Unit tests asserting majority-class and random-uniform baseline calculations
match theoretical closed-form values and empirical limits.
"""

import pytest
import numpy as np
from romanurdu.baselines import compute_majority_baseline, compute_uniform_baseline


def test_majority_baseline_synthetic():
    """Verify majority baseline on controlled distribution."""
    # Train: 80 'joy', 20 'sadness' -> Majority is 'joy'
    train_labels = ["joy"] * 80 + ["sadness"] * 20
    # Test: 60 'joy', 40 'sadness'
    test_labels = ["joy"] * 60 + ["sadness"] * 40

    res = compute_majority_baseline(train_labels, test_labels, num_classes=2)
    assert res["majority_label"] == "joy"
    # Accuracy should be exactly 60/100 = 0.60
    assert pytest.approx(res["accuracy"], abs=1e-5) == 0.60
    # For 'joy': precision = 60/100 = 0.60, recall = 60/60 = 1.0 -> F1 = 2*0.6*1.0 / 1.6 = 0.75
    # For 'sadness': F1 = 0
    # Macro-F1 = (0.75 + 0) / 2 = 0.375
    assert pytest.approx(res["f1_macro"], abs=1e-5) == 0.375


def test_uniform_baseline_closed_form():
    """Verify random-uniform theoretical floors match 1/C."""
    # 6 classes (Ekman-6)
    ekman_res = compute_uniform_baseline(
        test_labels=[0, 1, 2, 3, 4, 5] * 100,
        num_classes=6,
        num_simulations=50,
    )
    assert pytest.approx(ekman_res["theoretical_accuracy"], abs=1e-5) == 1.0 / 6.0
    assert pytest.approx(ekman_res["theoretical_f1_macro"], abs=1e-5) == 1.0 / 6.0
    # Simulation should be close to 1/6 (~0.1667)
    assert pytest.approx(ekman_res["simulated_accuracy"], abs=0.03) == 1.0 / 6.0

    # 28 classes (GoEmotions)
    goemo_res = compute_uniform_baseline(
        test_labels=list(range(28)) * 20,
        num_classes=28,
        num_simulations=50,
    )
    assert pytest.approx(goemo_res["theoretical_accuracy"], abs=1e-5) == 1.0 / 28.0
    assert pytest.approx(goemo_res["theoretical_f1_macro"], abs=1e-5) == 1.0 / 28.0
    assert pytest.approx(goemo_res["simulated_accuracy"], abs=0.02) == 1.0 / 28.0
