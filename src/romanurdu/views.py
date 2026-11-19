"""
Dataset view construction with k-controlled variant admission.

Provides the single shared build_view() entry point used by all tasks.
Controls variant admission via integer k (0=primary only, 1=primary+1 variant, 3=primary+3 variants).
Supports explicit length filtering (2-80 tokens) with configurable toggle.
"""

import re
from typing import Set, List, Union
import pandas as pd

_re_space = re.compile(r"\s+")
_re_punct = re.compile(r"([!?.,])\1+")


def normalize(text: str) -> str:
    """
    Standardize text: lowercase, collapse whitespace, cap repeated punctuation
    to maximum 2 chars, and strip non-printable characters.
    """
    if not isinstance(text, str):
        return ""
    t = text.strip().lower()
    t = _re_space.sub(" ", t)
    t = _re_punct.sub(r"\1\1", t)
    t = "".join(ch for ch in t if ch.isprintable())
    return t.strip()


def parse_variants(variants_str: Union[str, float, None]) -> List[str]:
    """Parse '||'-delimited string into list of trimmed variant strings."""
    if not isinstance(variants_str, str) or not variants_str.strip():
        return []
    return [v.strip() for v in variants_str.split("||") if v.strip()]


def build_view(
    df_source: pd.DataFrame,
    ids: Set[str],
    k: int = 0,
    apply_length_filter: bool = True,
    min_tokens: int = 2,
    max_tokens: int = 80,
) -> pd.DataFrame:
    """
    Construct a training-ready view dataframe from source data and a set of global_ids.

    Args:
        df_source: Source DataFrame containing 'global_id', 'label_str' (or 'label'),
                   'text_roman_ur', and optionally 'text_roman_ur_variants'.
        ids: Set of global_id strings belonging to this split.
        k: Number of spelling variants to admit per instance (0, 1, or 3).
        apply_length_filter: If True, filters texts to min_tokens <= len(split) <= max_tokens.
                             (True for primary 27-emotion task, False for standard Ekman-6 direct).
        min_tokens: Minimum whitespace-separated token count (default: 2).
        max_tokens: Maximum whitespace-separated token count (default: 80).

    Returns:
        DataFrame with columns ['global_id', 'text', 'label'] de-duplicated by (text, label).
    """
    if df_source.empty or not ids:
        return pd.DataFrame(columns=["global_id", "text", "label"])

    source_ids = set(str(i) for i in ids)
    df = df_source[df_source["global_id"].astype(str).isin(source_ids)].copy()

    # Determine label column
    label_col = "label_str" if "label_str" in df.columns else "label"

    rows = []
    for _, r in df.iterrows():
        gid = str(r["global_id"])
        lab = str(r[label_col]).strip().lower()

        primary = r.get("text_roman_ur", "")
        if isinstance(primary, str) and primary.strip():
            rows.append({"global_id": gid, "text": primary, "label": lab})

        if k > 0:
            var_col = r.get("text_roman_ur_variants", "")
            variants = parse_variants(var_col)
            for v in variants[:k]:
                rows.append({"global_id": gid, "text": v, "label": lab})

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["global_id", "text", "label"])

    # Normalization
    out["text"] = out["text"].apply(normalize)
    out = out[out["text"].str.len() > 0].reset_index(drop=True)

    # Optional length filtering
    if apply_length_filter:
        n_tokens = out["text"].str.split().apply(len)
        out = out[(n_tokens >= min_tokens) & (n_tokens <= max_tokens)].reset_index(drop=True)

    # De-duplicate identical (text, label) pairs
    out = out.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    return out[["global_id", "text", "label"]]
