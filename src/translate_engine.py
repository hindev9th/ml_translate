"""
Translation engine with three modes:

  1. Custom multilingual model  — single model trained from scratch,
                                  auto-detects language, uses >>lang<< tags
  2. CTranslate2 (INT8)         — fast per-pair models, ~80–120 MB each
  3. HuggingFace                — fallback, used during fine-tuning

Mode is selected automatically:
  - If models/custom_vi/ exists → use custom multilingual model
  - Else if CT2 model exists    → use CT2
  - Else                        → use HuggingFace
"""

import os
import logging
from collections import OrderedDict
from typing import List, Optional, Union

from .config import (
    LANGUAGE_PAIRS,
    PIVOT_MODELS,
    MODELS_DIR,
    get_pivot_ct2_path,
    get_pivot_ct2_exists,
)

CUSTOM_MODEL_DIR = os.path.join(MODELS_DIR, "custom_vi")
CUSTOM_CT2_DIR   = os.path.join(MODELS_DIR, "ct2_custom_vi")

logger = logging.getLogger(__name__)


class _ModelCache:
    """Simple LRU cache for loaded models."""

    def __init__(self, capacity: int = 3):
        self._capacity = capacity
        self._cache: OrderedDict = OrderedDict()

    def get(self, key: str):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, model):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._capacity:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug(f"Evicted model from cache: {evicted_key}")
            self._cache[key] = model

    def clear(self):
        self._cache.clear()


class _CT2Model:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = "cpu"):
        import ctranslate2
        from transformers import MarianTokenizer

        self.translator = ctranslate2.Translator(
            model_path,
            device=device,
            inter_threads=2,
            intra_threads=4,
        )
        self.tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)

    def translate(self, texts: List[str], beam_size: int = 4, max_length: int = 512) -> List[str]:
        tokenized = [self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(t, truncation=True, max_length=max_length)
        ) for t in texts]
        results = self.translator.translate_batch(
            tokenized,
            beam_size=beam_size,
            max_decoding_length=max_length,
        )
        decoded = [
            self.tokenizer.convert_tokens_to_string(r.hypotheses[0])
            for r in results
        ]
        return decoded


class _HFModel:
    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import MarianMTModel, MarianTokenizer
        import torch

        logger.info(f"Loading HF model: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def translate(self, texts: List[str], beam_size: int = 4, max_length: int = 512) -> List[str]:
        import torch

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=beam_size,
                max_length=max_length,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


class _MultilingualModel:
    """
    Wrapper for the custom single multilingual MarianMT model.
    Supports CTranslate2 (fast) and HuggingFace (fallback).
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self._ct2 = None
        self._hf  = None

        ct2_path = CUSTOM_CT2_DIR
        if os.path.isdir(ct2_path):
            import ctranslate2
            from transformers import MarianTokenizer
            self._ct2 = ctranslate2.Translator(
                ct2_path, device=device, inter_threads=2, intra_threads=4
            )
            self._tokenizer = MarianTokenizer.from_pretrained(model_path)
            logger.info(f"Loaded custom multilingual model (CT2 INT8): {ct2_path}")
        elif os.path.isdir(model_path):
            from transformers import MarianMTModel, MarianTokenizer
            self._tokenizer = MarianTokenizer.from_pretrained(model_path)
            self._hf_model  = MarianMTModel.from_pretrained(model_path)
            self._hf_model.to(device)
            self._hf_model.eval()
            self._device = device
            logger.info(f"Loaded custom multilingual model (HF): {model_path}")
        else:
            raise RuntimeError(
                f"Custom model not found at {model_path}.\n"
                "Train it first: python src/multilingual_train.py"
            )

    def translate(
        self,
        texts: List[str],
        tgt_lang: str,
        beam_size: int = 4,
        max_length: int = 512,
    ) -> List[str]:
        from .multilingual_tokenizer import add_language_tag

        tagged = [add_language_tag(t, tgt_lang) for t in texts]

        if self._ct2 is not None:
            tokenized = [
                self._tokenizer.convert_ids_to_tokens(
                    self._tokenizer.encode(t, truncation=True, max_length=max_length)
                )
                for t in tagged
            ]
            results = self._ct2.translate_batch(
                tokenized, beam_size=beam_size, max_decoding_length=max_length
            )
            return [
                self._tokenizer.convert_tokens_to_string(r.hypotheses[0])
                for r in results
            ]
        else:
            import torch
            inputs = self._tokenizer(
                tagged, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length
            ).to(self._device)
            with torch.no_grad():
                outputs = self._hf_model.generate(
                    **inputs, num_beams=beam_size, max_length=max_length
                )
            return self._tokenizer.batch_decode(outputs, skip_special_tokens=True)


class TranslationEngine:
    """
    Main translation engine — automatically selects the best available backend:

      1. Custom multilingual model (models/custom_vi/) — single model for all pairs
      2. CTranslate2 INT8 per-pair models             — fast, low RAM
      3. HuggingFace per-pair models                  — fallback

    Usage:
        engine = TranslationEngine()
        result = engine.translate("Hello world", src="en", tgt="vi")
        results = engine.translate_batch(["Hello", "World"], src="en", tgt="vi")
        result = engine.translate("你好", src="auto", tgt="vi")  # auto-detect
    """

    def __init__(self, device: str = "cpu", max_loaded_models: int = 3):
        self.device = device
        self._cache = _ModelCache(capacity=max_loaded_models)
        self._custom_model: Optional[_MultilingualModel] = None
        self._use_custom = os.path.isdir(CUSTOM_MODEL_DIR)
        if self._use_custom:
            logger.info("Custom multilingual model detected — will use it for all pairs.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        src: str,
        tgt: str,
        beam_size: int = 4,
        max_length: int = 512,
    ) -> str:
        results = self.translate_batch([text], src, tgt, beam_size, max_length)
        return results[0]

    def translate_batch(
        self,
        texts: List[str],
        src: str,
        tgt: str,
        beam_size: int = 4,
        max_length: int = 512,
    ) -> List[str]:
        # Auto-detect source language
        if src == "auto":
            src = detect_language(texts[0] if texts else "")

        pair_id = f"{src}-{tgt}"
        if pair_id not in LANGUAGE_PAIRS:
            raise ValueError(f"Unsupported language pair: {pair_id}")

        # Mode 1: custom multilingual model
        if self._use_custom:
            return self._translate_custom(texts, tgt, beam_size, max_length)

        # Mode 2/3: per-pair models
        cfg = LANGUAGE_PAIRS[pair_id]
        if cfg.use_pivot:
            return self._translate_pivot(texts, src, tgt, beam_size, max_length)
        return self._translate_direct(texts, pair_id, cfg, beam_size, max_length)

    def warmup(self, pair_ids: Optional[List[str]] = None):
        """Pre-load models into cache so first request is fast."""
        if self._use_custom:
            logger.info("Warming up custom multilingual model ...")
            self._get_custom_model()
            return
        if pair_ids is None:
            pair_ids = ["en-vi", "vi-en"]
        for pid in pair_ids:
            cfg = LANGUAGE_PAIRS.get(pid)
            if cfg and not cfg.use_pivot:
                self._load_model(pid, cfg)

    def clear_cache(self):
        self._cache.clear()
        self._custom_model = None

    # ------------------------------------------------------------------
    # Custom multilingual model
    # ------------------------------------------------------------------

    def _get_custom_model(self) -> _MultilingualModel:
        if self._custom_model is None:
            self._custom_model = _MultilingualModel(CUSTOM_MODEL_DIR, self.device)
        return self._custom_model

    def _translate_custom(self, texts, tgt_lang, beam_size, max_length):
        model = self._get_custom_model()
        return model.translate(texts, tgt_lang=tgt_lang, beam_size=beam_size, max_length=max_length)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _translate_direct(self, texts, pair_id, cfg, beam_size, max_length):
        model = self._load_model(pair_id, cfg)
        return model.translate(texts, beam_size=beam_size, max_length=max_length)

    def _translate_pivot(self, texts, src, tgt, beam_size, max_length):
        """Two-step translation via English: SRC→EN→TGT."""
        # Step 1: src → en
        src_en_id = f"{src}-en"
        mid_texts = self._translate_with_model_id(texts, src_en_id, beam_size, max_length)
        # Step 2: en → tgt
        en_tgt_id = f"en-{tgt}"
        return self._translate_with_model_id(mid_texts, en_tgt_id, beam_size, max_length)

    def _translate_with_model_id(self, texts, pair_id, beam_size, max_length):
        """Translate using a pivot model (may be direct or from PIVOT_MODELS registry)."""
        cached = self._cache.get(pair_id)
        if cached:
            return cached.translate(texts, beam_size=beam_size, max_length=max_length)

        # Try CTranslate2 converted pivot model first
        ct2_path = get_pivot_ct2_path(pair_id)
        if ct2_path and os.path.isdir(ct2_path):
            tokenizer_src = self._resolve_tokenizer_path(pair_id)
            model = _CT2Model(ct2_path, tokenizer_src, self.device)
            self._cache.put(pair_id, model)
            return model.translate(texts, beam_size=beam_size, max_length=max_length)

        # Check if it's a direct language pair with a CT2 model
        cfg = LANGUAGE_PAIRS.get(pair_id)
        if cfg:
            return self._translate_direct(texts, pair_id, cfg, beam_size, max_length)

        # Fall back to HF pivot model
        hf_name = PIVOT_MODELS.get(pair_id)
        if hf_name:
            model = _HFModel(hf_name, self.device)
            self._cache.put(pair_id, model)
            return model.translate(texts, beam_size=beam_size, max_length=max_length)

        raise ValueError(f"No model available for pivot pair: {pair_id}")

    def _load_model(self, pair_id, cfg):
        cached = self._cache.get(pair_id)
        if cached:
            return cached

        # Priority: fine-tuned CT2 > HF fine-tuned > base CT2 > base HF
        ft_ct2 = os.path.join(cfg.local_ct2_path + "_ft", "")
        if os.path.isdir(ft_ct2):
            tokenizer_src = self._resolve_tokenizer_path(pair_id, cfg=cfg)
            model = _CT2Model(ft_ct2, tokenizer_src, self.device)
        elif cfg.has_local_ct2():
            tokenizer_src = self._resolve_tokenizer_path(pair_id, cfg=cfg)
            model = _CT2Model(cfg.local_ct2_path, tokenizer_src, self.device)
        elif cfg.has_local_ft():
            model = _HFModel(cfg.local_ft_path, self.device)
        elif cfg.hf_model:
            model = _HFModel(cfg.hf_model, self.device)
        else:
            raise RuntimeError(f"No model found for {pair_id}. Run: python scripts/download_models.py --pair {pair_id}")

        self._cache.put(pair_id, model)
        return model

    def _resolve_tokenizer_path(self, pair_id: str, cfg=None) -> str:
        """Return the best tokenizer path for a pair."""
        if cfg and cfg.has_local_ft():
            return cfg.local_ft_path
        if cfg and cfg.hf_model:
            return cfg.hf_model
        hf_name = PIVOT_MODELS.get(pair_id)
        if hf_name:
            return hf_name
        raise ValueError(f"Cannot resolve tokenizer for {pair_id}")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_SCRIPT_MAP = {
    # CJK Unified Ideographs — Chinese (and some Japanese Kanji)
    (0x4E00, 0x9FFF): "zh",
    (0x3400, 0x4DBF): "zh",
    # Hiragana / Katakana — Japanese
    (0x3040, 0x30FF): "ja",
    # Hangul — Korean
    (0xAC00, 0xD7A3): "ko",
    (0x1100, 0x11FF): "ko",
}


def detect_language(text: str) -> str:
    """
    Fast script-based language detection — no external library needed.

    Priority order: JA > KO > ZH > VI/EN
    For VI vs EN we use langdetect if installed, otherwise default to EN.
    """
    if not text.strip():
        return "en"

    # Count characters per script
    counts = {"zh": 0, "ja": 0, "ko": 0}
    for ch in text:
        cp = ord(ch)
        for (lo, hi), lang in _SCRIPT_MAP.items():
            if lo <= cp <= hi:
                counts[lang] = counts.get(lang, 0) + 1

    total = len([c for c in text if not c.isspace()])
    if total == 0:
        return "en"

    # If >10% of characters are script-specific, classify
    for lang in ("ja", "ko", "zh"):   # JA first: Hiragana/Katakana are unambiguous
        if counts[lang] / total > 0.10:
            return lang

    # Latin script: try to distinguish EN vs VI
    try:
        from langdetect import detect as _ld
        detected = _ld(text)
        if detected in ("en", "vi"):
            return detected
        return "en"
    except Exception:
        pass

    # Heuristic: Vietnamese has many diacritics
    vi_chars = sum(1 for c in text if ord(c) > 0x00BF and c.isalpha())
    if vi_chars / max(total, 1) > 0.15:
        return "vi"

    return "en"


# ---------------------------------------------------------------------------
# CTranslate2 conversion for the custom model
# ---------------------------------------------------------------------------

def convert_custom_to_ct2(quantization: str = "int8"):
    """
    Convert the custom multilingual model to CTranslate2 format.
    Call this after training to get fast inference.
    """
    if not os.path.isdir(CUSTOM_MODEL_DIR):
        raise RuntimeError(f"Custom model not found at {CUSTOM_MODEL_DIR}")
    try:
        import ctranslate2
    except ImportError:
        raise ImportError("pip install ctranslate2")

    logger.info(f"Converting {CUSTOM_MODEL_DIR} → {CUSTOM_CT2_DIR} ({quantization}) ...")
    ctranslate2.converters.OpusMTConverter(CUSTOM_MODEL_DIR).convert(
        CUSTOM_CT2_DIR, quantization=quantization, force=True
    )
    logger.info(f"Done: {CUSTOM_CT2_DIR}")
