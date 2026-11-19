"""
Translation quality assurance and sanity check heuristics.

Flags potentially hallucinated, truncated, or over-sanitized Roman Urdu translations
produced during LLM generation. Identifies the 8,619 flagged instances analyzed in the paper.
"""

from typing import Optional, List

SUSPICIOUS_EXTRA_TOKENS: List[str] = [
    "pakistan",
    "india",
    "america",
    "usa",
    "uk",
    "karachi",
    "lahore",
    "islamabad",
]

PROFANITY_SOURCE_WORDS: List[str] = [
    "fuck",
    "fucking",
    "shit",
    "bastard",
]

PROFANITY_TARGET_WORDS: List[str] = [
    "fuck",
    "shit",
    "harami",
    "bhosdi",
    "bhen",
    "laanat",
    "lanat",
]


def check_translation_quality(src: str, tgt: str) -> Optional[str]:
    """
    Evaluate candidate translation against source English text for QA anomalies.

    Checks:
    - Empty or whitespace-only translation
    - Hallucinated geographical/entity tokens not present in source
    - Length ratio anomalies (ratio < 0.3 or ratio > 3.0)
    - Missing token boundaries (no spaces in a multi-word translation)
    - Sanitized profanity (source has profanity but target appears bleached)

    Args:
        src: English source text.
        tgt: Roman Urdu translated text.

    Returns:
        Reason string if flagged, or None if translation passes sanity checks.
    """
    if not isinstance(tgt, str) or not tgt.strip():
        return "empty_translation"

    if not isinstance(src, str):
        src = ""

    src_l = src.lower()
    tgt_l = tgt.lower()

    # Check 1: Added suspicious entities / hallucinated tokens
    for tok in SUSPICIOUS_EXTRA_TOKENS:
        if tok in tgt_l and tok not in src_l:
            return f"added_suspicious_token:{tok}"

    # Check 2: Length ratio check
    src_len = len(src_l)
    tgt_len = len(tgt_l)
    if src_len > 0:
        ratio = tgt_len / src_len
        if ratio < 0.3:
            return "too_short_vs_source"
        if ratio > 3.0:
            return "too_long_vs_source"

    # Check 3: Spaces in long sentence
    if len(src_l.split()) >= 5 and " " not in tgt.strip():
        return "no_spaces_in_long_sentence"

    # Check 4: Possible sanitized profanity
    if any(bad in src_l for bad in PROFANITY_SOURCE_WORDS):
        if not any(rude in tgt_l for rude in PROFANITY_TARGET_WORDS):
            return "possible_sanitized_profanity"

    return None
