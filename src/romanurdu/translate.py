"""
LLM-based English -> Roman Urdu translation pipeline using Ollama (gemma3:latest).
Handles single-line generation, timeout management, variant expansion, and quality flagging.
"""

import logging
import subprocess
from typing import List, Tuple, Optional
from .variants import generate_spelling_variants
from .qa_flagging import check_translation_quality

DEFAULT_OLLAMA_MODEL = "gemma3:latest"
DEFAULT_TIMEOUT = 10  # seconds


def get_primary_label(label_ids, go_label_names: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """Extract primary integer and string label from GoEmotions label IDs."""
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
    """
    Subprocess-based interface to local Ollama instance running Gemma 3.
    """

    def __init__(self, model_name: str = DEFAULT_OLLAMA_MODEL, timeout: int = DEFAULT_TIMEOUT):
        self.model_name = model_name
        self.timeout = timeout
        logging.info("GemmaRomanUrduTranslator initialized with model '%s' (timeout=%ds).", self.model_name, self.timeout)

    def _build_prompt(self, text: str) -> str:
        return (
            "You are a professional translator for an NLP research dataset.\n"
            "Translate English sentences into natural Pakistani Roman Urdu.\n"
            "Keep meaning, emotion, and tone. No added info.\n"
            "One sentence only.\n\n"
            f"English: {text}\nRoman Urdu:"
        )

    def translate_single(self, text: str) -> str:
        """Translate a single English string to Roman Urdu."""
        if not text or not text.strip():
            return ""
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
                timeout=self.timeout,
            )
            output = result.stdout.strip()
        except subprocess.TimeoutExpired:
            logging.warning("Translation timed out (>%ds) for text: %.30s...", self.timeout, text)
            return ""
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error("Ollama execution failed: %s", e)
            return ""

        lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
        return lines[0] if lines else ""


def convert_to_roman_urdu_with_variants(
    text_en: str,
    translator: GemmaRomanUrduTranslator,
    max_variants: int = 5,
) -> Tuple[str, List[str], Optional[str]]:
    """
    Translate English sentence, generate deterministic spelling variants,
    and compute QA flag status.
    """
    primary = translator.translate_single(text_en)
    variants = generate_spelling_variants(primary, max_variants=max_variants)
    flag_reason = check_translation_quality(text_en, primary)
    return primary, variants, flag_reason
