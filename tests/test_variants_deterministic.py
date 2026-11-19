"""
Unit tests for deterministic orthographic variation generation.
"""

import pytest
from romanurdu.variants import generate_spelling_variants


def test_variants_deterministic_repeated_calls():
    """Verify that identical input strings produce identical variants across calls."""
    sentence = "mujhe tum se bohot mohabbat hai aur main khush hoon"
    res1 = generate_spelling_variants(sentence, max_variants=5)
    res2 = generate_spelling_variants(sentence, max_variants=5)
    res3 = generate_spelling_variants(sentence, max_variants=5)

    assert res1 == res2 == res3
    assert len(res1) <= 5
    # Ensure all variants are non-empty and distinct
    assert len(set(res1)) == len(res1)
    assert sentence not in res1


def test_variants_stable_alphabetical_order():
    """Verify that returned variants are sorted alphabetically to prevent set-order jitter."""
    sentence = "kya haal hai bhai jaan ghabrana nahi hai"
    variants = generate_spelling_variants(sentence, max_variants=5)
    assert variants == sorted(variants)


def test_variants_empty_and_whitespace():
    """Verify edge cases with empty or whitespace-only inputs."""
    assert generate_spelling_variants("") == []
    assert generate_spelling_variants("   ") == []
    assert generate_spelling_variants(None) == []


def test_variants_rule_coverage():
    """Verify phonetic transformation rules trigger properly."""
    # kh -> x
    v_kh = generate_spelling_variants("khabar")
    assert any("xabar" in v for v in v_kh)

    # x -> kh
    v_x = generate_spelling_variants("xabar")
    assert any("khabar" in v for v in v_x)

    # oo -> u and u -> oo
    v_oo = generate_spelling_variants("soorat")
    assert any("surat" in v for v in v_oo)
