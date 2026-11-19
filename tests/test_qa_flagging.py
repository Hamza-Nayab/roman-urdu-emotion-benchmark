"""
Unit tests for translation quality assurance and sanity check heuristics.
"""

import pytest
from romanurdu.qa_flagging import check_translation_quality


def test_empty_translation():
    assert check_translation_quality("Hello world", "") == "empty_translation"
    assert check_translation_quality("Hello world", "   ") == "empty_translation"
    assert check_translation_quality("Hello world", None) == "empty_translation"


def test_hallucinated_tokens():
    # 'karachi' added in target but not in source
    res = check_translation_quality("I am going home today", "main karachi ghar ja raha hoon")
    assert res == "added_suspicious_token:karachi"

    # 'karachi' present in both source and target -> should NOT be flagged as added
    res_ok = check_translation_quality("I love Karachi city", "mujhe karachi shehar pasand hai")
    assert res_ok is None


def test_length_ratios():
    # Target way too short (< 0.3 ratio)
    src = "This is a very long and detailed English sentence that should not become one syllable"
    tgt_short = "haan"
    assert check_translation_quality(src, tgt_short) == "too_short_vs_source"

    # Target way too long (> 3.0 ratio)
    src_short = "Yes"
    tgt_long = "haan bilkul sahi kaha aap ne main aap ki baat se poora raazi hoon"
    assert check_translation_quality(src_short, tgt_long) == "too_long_vs_source"


def test_no_spaces_in_long_sentence():
    src = "one two three four five"
    tgt = "onetwothreefourfive"
    assert check_translation_quality(src, tgt) == "no_spaces_in_long_sentence"


def test_sanitized_profanity():
    # Source has profanity, target translates neutrally without preserving vulgarity
    src = "This is fucking terrible"
    tgt_clean = "yeh bohot bura hai"
    assert check_translation_quality(src, tgt_clean) == "possible_sanitized_profanity"

    # Source has profanity, target retains appropriate vulgarity
    tgt_vulgar = "yeh harami pan hai"
    assert check_translation_quality(src, tgt_vulgar) is None
