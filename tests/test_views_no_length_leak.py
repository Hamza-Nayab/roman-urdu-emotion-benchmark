"""
Unit tests asserting view construction properties:
1. When apply_length_filter=True, token length is strictly between 2 and 80.
2. Variant admission is strictly governed by k in {0, 1, 3}.
"""

import pytest
import pandas as pd
from romanurdu.views import build_view, normalize, parse_variants


def test_normalize_function():
    """Verify whitespace collapsing, punctuation repeat capping, and lowercasing."""
    raw = "  HaAn   Bhai!!!!  Kaisey    ho???  "
    norm = normalize(raw)
    assert norm == "haan bhai!! kaisey ho??"


def test_parse_variants():
    """Verify variant string splitting on '||'."""
    raw = "var_one || var_two ||  var_three  "
    parsed = parse_variants(raw)
    assert parsed == ["var_one", "var_two", "var_three"]
    assert parse_variants("") == []
    assert parse_variants(None) == []


def test_view_k_admission_counts():
    """Verify that k=0 admits only primary, k=1 admits 1 variant, k=3 admits 3 variants."""
    source_df = pd.DataFrame([
        {
            "global_id": "test_000001",
            "label_str": "joy",
            "text_roman_ur": "yeh bohot achi khabar hai",
            "text_roman_ur_variants": "yeh bohot achi xabar hai || yeh bht achi khabar hai || yeh bohot achii khabar hai || extra variant",
        },
        {
            "global_id": "test_000002",
            "label_str": "sadness",
            "text_roman_ur": "main bohot udaas hoon",
            "text_roman_ur_variants": "main bht udaas hoon || main bohot udas hoon",
        },
    ])
    ids = {"test_000001", "test_000002"}

    # k = 0 (primary only)
    view_k0 = build_view(source_df, ids, k=0, apply_length_filter=False)
    assert len(view_k0) == 2

    # k = 1 (primary + 1 variant each)
    view_k1 = build_view(source_df, ids, k=1, apply_length_filter=False)
    assert len(view_k1) == 4

    # k = 3 (instance 1 has 1 primary + 3 variants = 4; instance 2 has 1 primary + 2 variants = 3; total = 7)
    view_k3 = build_view(source_df, ids, k=3, apply_length_filter=False)
    assert len(view_k3) == 7


def test_view_length_filter_bounds():
    """Verify that apply_length_filter=True removes texts with <2 tokens or >80 tokens."""
    source_df = pd.DataFrame([
        {"global_id": "id_1", "label": "joy", "text_roman_ur": "shukriya"},  # 1 token -> should be filtered
        {"global_id": "id_2", "label": "joy", "text_roman_ur": "bohot shukriya"},  # 2 tokens -> valid
        {"global_id": "id_3", "label": "joy", "text_roman_ur": " ".join(["lafz"] * 80)},  # 80 tokens -> valid
        {"global_id": "id_4", "label": "joy", "text_roman_ur": " ".join(["lafz"] * 81)},  # 81 tokens -> should be filtered
    ])
    ids = {"id_1", "id_2", "id_3", "id_4"}

    filtered_view = build_view(source_df, ids, k=0, apply_length_filter=True, min_tokens=2, max_tokens=80)
    assert len(filtered_view) == 2
    assert set(filtered_view["global_id"]) == {"id_2", "id_3"}

    # When apply_length_filter=False, 1-token and 81-token texts are retained
    unfiltered_view = build_view(source_df, ids, k=0, apply_length_filter=False)
    assert len(unfiltered_view) == 4
