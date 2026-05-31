"""
Language pair configurations and model registry.
Supports direct translation (if model exists) and pivot through English.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict

# Models directory (local fine-tuned / CTranslate2 converted)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

LANG_NAMES = {
    "en": "English",
    "vi": "Tiếng Việt",
    "zh": "中文",
    "ja": "日本語",
    "ko": "한국어",
}

LANG_FLAGS = {
    "en": "🇬🇧",
    "vi": "🇻🇳",
    "zh": "🇨🇳",
    "ja": "🇯🇵",
    "ko": "🇰🇷",
}


@dataclass
class LangPairConfig:
    src: str
    tgt: str
    # HuggingFace model ID (for download + fine-tuning base)
    hf_model: Optional[str] = None
    # Pivot language when no direct model exists
    pivot_lang: Optional[str] = None

    @property
    def pair_id(self) -> str:
        return f"{self.src}-{self.tgt}"

    @property
    def use_pivot(self) -> bool:
        return self.hf_model is None

    @property
    def local_ct2_path(self) -> str:
        return os.path.join(MODELS_DIR, f"ct2_{self.pair_id}")

    @property
    def local_ft_path(self) -> str:
        return os.path.join(MODELS_DIR, f"ft_{self.pair_id}")

    def has_local_ct2(self) -> bool:
        return os.path.isdir(self.local_ct2_path)

    def has_local_ft(self) -> bool:
        return os.path.isdir(self.local_ft_path)


# Direct HuggingFace models for each pair
# ZH/JA/KO ↔ VI use pivot through English (no direct OPUS-MT model available)
LANGUAGE_PAIRS: Dict[str, LangPairConfig] = {
    "en-vi": LangPairConfig("en", "vi", hf_model="Helsinki-NLP/opus-mt-en-vi"),
    "vi-en": LangPairConfig("vi", "en", hf_model="Helsinki-NLP/opus-mt-vi-en"),
    "zh-vi": LangPairConfig("zh", "vi", pivot_lang="en"),
    "vi-zh": LangPairConfig("vi", "zh", pivot_lang="en"),
    "ja-vi": LangPairConfig("ja", "vi", pivot_lang="en"),
    "vi-ja": LangPairConfig("vi", "ja", pivot_lang="en"),
    "ko-vi": LangPairConfig("ko", "vi", pivot_lang="en"),
    "vi-ko": LangPairConfig("vi", "ko", pivot_lang="en"),
}

# Intermediate models used for pivot translation
PIVOT_MODELS: Dict[str, str] = {
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "ja-en": "Helsinki-NLP/opus-mt-jap-en",
    "en-ja": "Helsinki-NLP/opus-mt-en-jap",
    "ko-en": "Helsinki-NLP/opus-mt-ko-en",
    "en-ko": "Helsinki-NLP/opus-mt-en-ko",
}

# CT2 local paths for pivot models
PIVOT_CT2_PATHS: Dict[str, str] = {
    k: os.path.join(MODELS_DIR, f"ct2_{k}") for k in PIVOT_MODELS
}


def get_pivot_ct2_path(pair_id: str) -> str:
    return PIVOT_CT2_PATHS.get(pair_id, "")


def get_pivot_ct2_exists(pair_id: str) -> bool:
    return os.path.isdir(get_pivot_ct2_path(pair_id))
