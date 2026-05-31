"""
Train a shared SentencePiece tokenizer for all 5 languages (EN, ZH, JA, KO, VI).

Design choices:
- BPE model type (handles CJK and Latin well)
- character_coverage=0.9999  (captures rare CJK/Vietnamese characters)
- vocab_size=50000           (large enough for 5 very different scripts)
- Language prefix tokens added as user-defined symbols: >>vi<<, >>en<<, etc.
"""

import os
import logging
import tempfile
from typing import List, Optional

logger = logging.getLogger(__name__)

LANG_CODES = ["en", "vi", "zh", "ja", "ko"]
LANG_TOKENS = [f">>{c}<<" for c in LANG_CODES]

# Default paths
_HERE = os.path.dirname(os.path.dirname(__file__))
DEFAULT_SPM_PREFIX = os.path.join(_HERE, "models", "custom_spm")
DEFAULT_SPM_MODEL  = DEFAULT_SPM_PREFIX + ".model"
DEFAULT_SPM_VOCAB  = DEFAULT_SPM_PREFIX + ".vocab"


def train_tokenizer(
    corpus_texts: List[str],
    model_prefix: str = DEFAULT_SPM_PREFIX,
    vocab_size: int = 50_000,
    character_coverage: float = 0.9999,
    num_threads: int = 4,
    input_sentence_size: int = 5_000_000,
    shuffle: bool = True,
):
    """
    Train a shared SentencePiece BPE tokenizer on a combined corpus.

    corpus_texts  : list of raw strings (all languages mixed)
    model_prefix  : output path prefix (.model + .vocab will be created)
    vocab_size    : recommended 50K for 5 languages
    """
    import sentencepiece as spm

    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    # Write corpus to temp file
    tmp = model_prefix + "_tmp_corpus.txt"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in corpus_texts:
            line = line.strip()
            if line:
                f.write(line + "\n")

    logger.info(f"Training SentencePiece on {len(corpus_texts):,} lines → {model_prefix}")

    spm.SentencePieceTrainer.train(
        input=tmp,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="bpe",
        pad_id=1,
        unk_id=0,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        # Language prefix tokens as user-defined symbols (never split)
        user_defined_symbols=",".join(LANG_TOKENS),
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle,
        num_threads=num_threads,
        train_extremely_large_corpus=(len(corpus_texts) > 2_000_000),
    )
    os.remove(tmp)
    logger.info(f"Tokenizer saved: {model_prefix}.model  vocab={vocab_size}")
    return model_prefix + ".model"


def load_tokenizer(spm_model_path: str = DEFAULT_SPM_MODEL):
    """
    Load a trained SentencePiece model into a HuggingFace MarianTokenizer wrapper.
    Creates a minimal HF tokenizer config so it's compatible with MarianMTModel.
    """
    from transformers import MarianTokenizer
    import json

    spm_dir = os.path.dirname(spm_model_path)
    dest = os.path.join(spm_dir, "custom_tokenizer")
    os.makedirs(dest, exist_ok=True)

    # Copy the spm model to the expected filename
    import shutil
    shutil.copy(spm_model_path, os.path.join(dest, "source.spm"))
    shutil.copy(spm_model_path, os.path.join(dest, "target.spm"))

    # Load vocab from spm and build the token→id mapping
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model_path)
    vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}

    # Write vocab.json
    with open(os.path.join(dest, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # Minimal tokenizer_config.json
    config = {
        "tokenizer_class": "MarianTokenizer",
        "model_max_length": 512,
        "source_lang": None,
        "target_lang": None,
    }
    with open(os.path.join(dest, "tokenizer_config.json"), "w") as f:
        json.dump(config, f)

    tokenizer = MarianTokenizer(
        vocab=os.path.join(dest, "vocab.json"),
        source_spm=os.path.join(dest, "source.spm"),
        target_spm=os.path.join(dest, "target.spm"),
    )
    return tokenizer


def load_spm_tokenizer(spm_model_path: str = DEFAULT_SPM_MODEL):
    """
    Return a raw SentencePieceProcessor (simpler, for data preprocessing).
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(spm_model_path)
    return sp


def add_language_tag(text: str, tgt_lang: str) -> str:
    """Prepend MarianMT-style language tag: >>vi<< Hello → >>vi<< Hello"""
    tag = f">>{tgt_lang}<<"
    if text.startswith(tag):
        return text
    return f"{tag} {text}"
