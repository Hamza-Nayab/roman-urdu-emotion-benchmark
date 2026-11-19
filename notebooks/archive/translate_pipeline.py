# ARCHIVED FOR PROVENANCE ONLY. Superseded by scripts/02_translate.py and src/romanurdu/translate.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Research-grade, incremental English → Roman Urdu conversion pipeline
for the HuggingFace GoEmotions dataset ("simplified" config).
Automatically processes in chunks of 100 rows.
Skips any single translation taking more than 10 seconds.
"""
import argparse
import logging
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

DATASET_NAME = "go_emotions"
DATASET_CONFIG = "simplified"
SOURCE_CSV = Path("goemotions_en_all_firstlabel.csv")
ROMAN_URDU_CSV = Path("goemotions_roman_ur_firstlabel.csv")
FLAGGED_CSV = Path("goemotions_roman_ur_flagged.csv")
DEFAULT_OLLAMA_MODEL = "gemma3:latest"
CHUNK_SIZE = 100
OLLAMA_TIMEOUT = 10

def get_primary_label(label_ids, go_label_names: List[str]) -> Tuple[Optional[int], Optional[str]]:
    if label_ids is None:
        return None, None
    try:
        ids = list(label_ids)
    except TypeError:
        return None, None
    if len(ids) == 0:
        return None, None
    lid = int(ids[0])
    if lid < 0 or lid >= len(go_label_names):
        return None, None
    return lid, go_label_names[lid]

class GemmaRomanUrduTranslator:
    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL):
        self.model_name = model_name
        logging.info(f"Using Ollama model '{model_name}' for EN→Roman Urdu translation.")

    def _build_prompt(self, text: str) -> str:
        return (
            "You are a professional translator for an NLP research dataset.\n"
            "Translate English sentences into natural Pakistani Roman Urdu.\n"
            "Keep meaning, emotion, and tone. No added info.\n"
            "One sentence only.\n\n"
            f"English: {text}\nRoman Urdu:"
        )

    def translate_single(self, text: str) -> str:
        prompt = self._build_prompt(text)
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=True,
                timeout=10
            )
            output = result.stdout.strip()
        except subprocess.TimeoutExpired:
            logging.warning("Timeout: skipping slow line (>10s)")
            return ""
        except subprocess.CalledProcessError as e:
            logging.error("Ollama call failed: %s", e.stderr)
            return ""
        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        return lines[0] if lines else ""

def generate_spelling_variants(primary: str, max_variants: int = 5) -> List[str]:
    import re
    variants = set()
    if "kh" in primary: variants.add(primary.replace("kh", "x"))
    if "x" in primary: variants.add(primary.replace("x", "kh"))
    variants.add(re.sub(r"aa+", "a", primary))
    variants.add(re.sub(r"a", "aa", primary, count=1))
    if "oo" in primary: variants.add(primary.replace("oo", "u"))
    variants.add(primary.replace("u", "oo"))
    variants.add(primary.replace("eh", "e"))
    variants.add(primary.replace("e", "eh"))
    variants.add(primary.replace("gh", "g"))
    variants.add(primary.replace("g", "gh"))
    cleaned = { " ".join(v.split()) for v in variants if v and v != primary }
    return list(cleaned)[:max_variants]

def convert_to_roman_urdu_with_variants(text_en: str, translator: GemmaRomanUrduTranslator, max_variants: int = 5):
    primary = translator.translate_single(text_en)
    variants = generate_spelling_variants(primary, max_variants=max_variants)
    return primary, variants

SUSPICIOUS_EXTRA_TOKENS = ["pakistan","india","america","usa","uk","karachi","lahore","islamabad"]

def check_translation_quality(src: str, tgt: str) -> Optional[str]:
    if not tgt or not tgt.strip():
        return "empty_translation"
    src_l = src.lower()
    tgt_l = tgt.lower()
    for tok in SUSPICIOUS_EXTRA_TOKENS:
        if tok in tgt_l and tok not in src_l:
            return f"added_suspicious_token:{tok}"
    src_len = len(src_l)
    tgt_len = len(tgt_l)
    if src_len > 0:
        ratio = tgt_len / src_len
        if ratio < 0.3: return "too_short_vs_source"
        if ratio > 3.0: return "too_long_vs_source"
    if len(src_l.split()) >= 5 and " " not in tgt.strip():
        return "no_spaces_in_long_sentence"
    if any(bad in src_l for bad in ["fuck", "fucking", "shit", "bastard"]):
        if not any(rude in tgt_l for rude in ["fuck", "shit", "harami", "bhosdi", "bhen", "laanat", "lanat"]):
            return "possible_sanitized_profanity"
    return None
