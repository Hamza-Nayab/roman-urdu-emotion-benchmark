"""
Roman Urdu Emotion Benchmark (romanurdu) package.

Provides reproducible dataset construction, deterministic orthographic variation
modeling, Ekman-6 standardization, unified model training, and baseline evaluation.
"""

from .variants import generate_spelling_variants
from .qa_flagging import check_translation_quality, SUSPICIOUS_EXTRA_TOKENS
from .translate import GemmaRomanUrduTranslator
from .splits import create_canonical_splits, load_split_ids, CANONICAL_SPLIT_COUNTS
from .views import build_view, normalize, parse_variants
from .ekman import GOEMO_TO_EKMAN, EKMAN_LABELS, to_ekman
from .baselines import compute_majority_baseline, compute_uniform_baseline

__version__ = "1.0.0"
__all__ = [
    "generate_spelling_variants",
    "check_translation_quality",
    "SUSPICIOUS_EXTRA_TOKENS",
    "GemmaRomanUrduTranslator",
    "create_canonical_splits",
    "load_split_ids",
    "CANONICAL_SPLIT_COUNTS",
    "build_view",
    "normalize",
    "parse_variants",
    "GOEMO_TO_EKMAN",
    "EKMAN_LABELS",
    "to_ekman",
    "compute_majority_baseline",
    "compute_uniform_baseline",
]
