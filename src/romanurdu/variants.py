"""
Deterministic orthographic variation generation for Roman Urdu.

Fixes the non-determinism defect in earlier pipelines where Python set iteration
order caused arbitrary variant truncation depending on PYTHONHASHSEED.
Candidates are strictly sorted deterministically before truncating to max_variants.
"""

import re
from typing import List, Optional


def generate_spelling_variants(
    primary: str,
    max_variants: int = 5,
    sort_strategy: str = "alphabetical",
) -> List[str]:
    """
    Generate up to `max_variants` deterministic phonetic and orthographic variants
    of a primary Roman Urdu text string.

    Rules:
    - kh <-> x
    - aa+ -> a and a -> aa (first occurrence)
    - oo <-> u
    - eh <-> e
    - gh <-> g

    Args:
        primary: Primary Roman Urdu text string.
        max_variants: Maximum number of spelling variants to return (default: 5).
        sort_strategy: Sorting strategy to guarantee deterministic output.
                       Supported: 'alphabetical' (default), 'length_first'.

    Returns:
        A list of distinct, normalized spelling variants, deterministically ordered.
    """
    if not isinstance(primary, str) or not primary.strip():
        return []

    primary_clean = " ".join(primary.strip().split())
    candidates = set()

    # Rule 1: kh <-> x
    if "kh" in primary_clean:
        candidates.add(primary_clean.replace("kh", "x"))
    if "x" in primary_clean:
        candidates.add(primary_clean.replace("x", "kh"))

    # Rule 2: aa+ -> a and a -> aa
    candidates.add(re.sub(r"aa+", "a", primary_clean))
    candidates.add(re.sub(r"a", "aa", primary_clean, count=1))

    # Rule 3: oo <-> u
    if "oo" in primary_clean:
        candidates.add(primary_clean.replace("oo", "u"))
    if "u" in primary_clean:
        candidates.add(primary_clean.replace("u", "oo"))

    # Rule 4: eh <-> e
    if "eh" in primary_clean:
        candidates.add(primary_clean.replace("eh", "e"))
    if "e" in primary_clean:
        candidates.add(primary_clean.replace("e", "eh"))

    # Rule 5: gh <-> g
    if "gh" in primary_clean:
        candidates.add(primary_clean.replace("gh", "g"))
    if "g" in primary_clean:
        candidates.add(primary_clean.replace("g", "gh"))

    # Clean and filter out empty strings and strings identical to primary
    cleaned = {
        " ".join(v.split())
        for v in candidates
        if v and " ".join(v.split()) != primary_clean
    }

    # Deterministic sorting (CRITICAL: removes Python hash seed dependency)
    if sort_strategy == "length_first":
        sorted_variants = sorted(cleaned, key=lambda s: (len(s), s))
    else:
        # Standard alphabetical sort
        sorted_variants = sorted(cleaned)

    return sorted_variants[:max_variants]
