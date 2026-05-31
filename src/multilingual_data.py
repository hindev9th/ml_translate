"""
Data pipeline for multilingual MarianMT training.

Handles:
  - Downloading parallel corpora for all 5 language pairs
  - Adding language prefix tags (>>vi<<, >>en<<, ...)
  - Temperature-based language sampling to balance pairs
  - Building HuggingFace DatasetDict for Seq2SeqTrainer

Language pairs trained simultaneously:
  en↔vi, zh↔vi, ja↔vi, ko↔vi  (both directions = 8 sub-datasets)
  Data for ZH/JA/KO ↔ VI:
    - ZH-VI: CCMatrix / WikiMatrix (if available)
    - JA-VI: CCMatrix / WikiMatrix
    - KO-VI: CCMatrix / WikiMatrix
    - Fallback: generated via existing translate engine (back-translation)
"""

import os
import logging
import random
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# OPUS-100 dataset has direct en-vi, zh-en, ja-en, ko-en
# For ZH/JA/KO ↔ VI we try WikiMatrix (has many Asian language pairs)
# OPUS-100: high-quality EN-paired data
OPUS_DIRECT = {
    "en-vi": ("Helsinki-NLP/opus-100", "en-vi"),
    "zh-en": ("Helsinki-NLP/opus-100", "zh-en"),
    "ja-en": ("Helsinki-NLP/opus-100", "ja-en"),
    "ko-en": ("Helsinki-NLP/opus-100", "ko-en"),
}

DEFAULT_TEMPERATURE = 0.7


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_opus(pair_id: str, max_samples: int) -> Tuple[List[str], List[str]]:
    from datasets import load_dataset

    name, hf_name = OPUS_DIRECT[pair_id]
    src_code, tgt_code = pair_id.split("-")
    ds = load_dataset(name, hf_name, split="train")
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    sources = [item["translation"][src_code] for item in ds]
    targets = [item["translation"][tgt_code] for item in ds]
    return sources, targets


DATA_CACHE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache")

# WikiMatrix on OPUS POUTA servers — stable, no HF scripts needed
# URL: https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/{pair}.txt.zip
# Files inside zip: WikiMatrix.{pair}.{lang}
OPUS_POUTA = {
    "zh-vi": ("WikiMatrix", "vi-zh", "zh", "vi"),   # alphabetical: vi < zh
    "ja-vi": ("WikiMatrix", "ja-vi", "ja", "vi"),   # alphabetical: ja < vi
    "ko-vi": ("WikiMatrix", "ko-vi", "ko", "vi"),   # alphabetical: ko < vi
    "vi-zh": ("WikiMatrix", "vi-zh", "vi", "zh"),
    "vi-ja": ("WikiMatrix", "ja-vi", "vi", "ja"),
    "vi-ko": ("WikiMatrix", "ko-vi", "vi", "ko"),
}


def _download_opus_pouta(pair_id: str, max_samples: int) -> Tuple[List[str], List[str]]:
    """
    Tải trực tiếp từ OPUS POUTA servers qua HTTP (không cần HF dataset scripts).
    WikiMatrix: Wikipedia-based, ~300K–600K pairs cho CJK↔VI, chất lượng tốt.
    """
    import requests, zipfile, io

    if pair_id not in OPUS_POUTA:
        return [], []

    corpus, pair_name, src_lang, tgt_lang = OPUS_POUTA[pair_id]
    os.makedirs(DATA_CACHE, exist_ok=True)

    src_cache = os.path.join(DATA_CACHE, f"{corpus}.{pair_name}.{src_lang}")
    tgt_cache = os.path.join(DATA_CACHE, f"{corpus}.{pair_name}.{tgt_lang}")

    if not (os.path.exists(src_cache) and os.path.exists(tgt_cache)):
        url = f"https://object.pouta.csc.fi/OPUS-{corpus}/v1/moses/{pair_name}.txt.zip"
        logger.info(f"Downloading {corpus} {pair_name} from OPUS ({url}) ...")
        try:
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()
            buf = io.BytesIO()
            for chunk in r.iter_content(chunk_size=65536):
                buf.write(chunk)
            buf.seek(0)
            with zipfile.ZipFile(buf) as zf:
                names = zf.namelist()
                # Files inside: WikiMatrix.vi-zh.vi, WikiMatrix.vi-zh.zh
                def _extract(lang):
                    for n in names:
                        if n.endswith(f".{lang}"):
                            return zf.read(n).decode("utf-8").splitlines()
                    return []
                srcs = _extract(src_lang)
                tgts = _extract(tgt_lang)
            with open(src_cache, "w", encoding="utf-8") as f:
                f.write("\n".join(srcs))
            with open(tgt_cache, "w", encoding="utf-8") as f:
                f.write("\n".join(tgts))
            logger.info(f"Cached {corpus} {pair_name}: {len(srcs):,} pairs → {DATA_CACHE}")
        except Exception as e:
            logger.warning(f"OPUS POUTA download failed for {pair_name}: {e}")
            return [], []

    with open(src_cache, encoding="utf-8") as f:
        sources = f.read().splitlines()
    with open(tgt_cache, encoding="utf-8") as f:
        targets = f.read().splitlines()

    if max_samples and len(sources) > max_samples:
        sources = sources[:max_samples]
        targets = targets[:max_samples]

    logger.info(f"OPUS {pair_id}: {len(sources):,} pairs")
    return sources, targets


def _download_via_opustools(src_lang: str, tgt_lang: str, corpus: str, max_samples: int) -> Tuple[List[str], List[str]]:
    """Fallback: dùng opustools nếu được cài (pip install opustools)."""
    try:
        from opustools import OpusRead
    except ImportError:
        return [], []

    os.makedirs(DATA_CACHE, exist_ok=True)
    src_file = os.path.join(DATA_CACHE, f"{corpus}_{src_lang}-{tgt_lang}.{src_lang}")
    tgt_file = os.path.join(DATA_CACHE, f"{corpus}_{src_lang}-{tgt_lang}.{tgt_lang}")

    if not (os.path.exists(src_file) and os.path.exists(tgt_file)):
        logger.info(f"Downloading {corpus} {src_lang}-{tgt_lang} via opustools ...")
        try:
            reader = OpusRead(
                directory=corpus, source=src_lang, target=tgt_lang,
                write=[src_file, tgt_file], write_mode="moses",
                suppress_prompts=True,
            )
            reader.printPairs()
        except Exception as e:
            logger.warning(f"opustools download failed: {e}")
            return [], []

    with open(src_file, encoding="utf-8") as f:
        sources = f.read().splitlines()
    with open(tgt_file, encoding="utf-8") as f:
        targets = f.read().splitlines()

    if max_samples:
        sources, targets = sources[:max_samples], targets[:max_samples]
    return sources, targets


def download_pair(
    src_lang: str,
    tgt_lang: str,
    max_samples: int = 200_000,
) -> Tuple[List[str], List[str]]:
    """
    Download parallel data for src_lang → tgt_lang.
    Priority: OPUS-100 → NLLB → FLORES (fallback small)
    """
    pair_id = f"{src_lang}-{tgt_lang}"

    # 1. OPUS-100 (direct pair)
    if pair_id in OPUS_DIRECT:
        logger.info(f"Downloading OPUS-100 {pair_id} (max={max_samples:,}) ...")
        return _download_opus(pair_id, max_samples)

    # 2. OPUS-100 reversed
    rev_id = f"{tgt_lang}-{src_lang}"
    if rev_id in OPUS_DIRECT:
        logger.info(f"Downloading OPUS-100 {rev_id} (reversed) ...")
        t, s = _download_opus(rev_id, max_samples)
        return s, t

    # 3. OPUS POUTA direct HTTP download (WikiMatrix — CJK↔VI)
    srcs, tgts = _download_opus_pouta(pair_id, max_samples)
    if srcs:
        return srcs, tgts

    # 4. opustools fallback
    srcs, tgts = _download_via_opustools(src_lang, tgt_lang, "WikiMatrix", max_samples)
    if srcs:
        return srcs, tgts

    logger.warning(f"No data found for {pair_id}")
    return [], []


# ---------------------------------------------------------------------------
# Language tag helpers
# ---------------------------------------------------------------------------

def tag_source(text: str, tgt_lang: str) -> str:
    """Prepend >>LANG<< tag to source text (MarianMT convention)."""
    return f">>{tgt_lang}<< {text}"


def prepare_pairs(
    sources: List[str],
    targets: List[str],
    src_lang: str,
    tgt_lang: str,
    add_reverse: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Tag sources with >>tgt_lang<< and optionally add reverse direction.
    Returns (tagged_sources, targets).
    """
    from .data_utils import filter_pairs

    sources, targets = filter_pairs(sources, targets)

    tagged_src  = [tag_source(s, tgt_lang) for s in sources]
    tagged_tgt  = list(targets)

    if add_reverse:
        rev_src = [tag_source(t, src_lang) for t in targets]
        tagged_src.extend(rev_src)
        tagged_tgt.extend(sources)

    return tagged_src, tagged_tgt


# ---------------------------------------------------------------------------
# Multi-pair dataset builder
# ---------------------------------------------------------------------------

def build_multilingual_corpus(
    pair_configs: Dict[str, int],
    add_reverse: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Download and combine multiple language pairs with temperature sampling.

    pair_configs : {"en-vi": max_samples, "zh-vi": max_samples, ...}
    temperature  : sampling temperature (0.7 default, higher=more uniform)
    add_reverse  : also add tgt→src direction for bidirectional model

    Returns (all_sources, all_targets) — tagged and shuffled.
    """
    import math

    all_pools: Dict[str, Tuple[List[str], List[str]]] = {}

    for pair_id, max_n in pair_configs.items():
        src_lang, tgt_lang = pair_id.split("-")
        srcs, tgts = download_pair(src_lang, tgt_lang, max_samples=max_n)
        if srcs:
            tagged_s, tagged_t = prepare_pairs(srcs, tgts, src_lang, tgt_lang, add_reverse)
            all_pools[pair_id] = (tagged_s, tagged_t)
            logger.info(f"{pair_id}: {len(tagged_s):,} pairs")
        else:
            logger.warning(f"No data found for {pair_id}")

    if not all_pools:
        raise ValueError("No parallel data could be downloaded for any pair.")

    # Temperature sampling: oversample small corpora
    sizes = {k: len(v[0]) for k, v in all_pools.items()}
    total = sum(sizes.values())

    # Compute sampling probs
    raw_probs = {k: (s / total) ** (1 / temperature) for k, s in sizes.items()}
    norm = sum(raw_probs.values())
    probs = {k: v / norm for k, v in raw_probs.items()}

    target_total = sum(sizes.values())

    rng = random.Random(seed)
    combined_src, combined_tgt = [], []

    for pair_id, (srcs, tgts) in all_pools.items():
        n = max(1, round(probs[pair_id] * target_total))
        if n >= len(srcs):
            combined_src.extend(srcs)
            combined_tgt.extend(tgts)
        else:
            idx = rng.sample(range(len(srcs)), n)
            combined_src.extend(srcs[i] for i in idx)
            combined_tgt.extend(tgts[i] for i in idx)

    # Shuffle combined
    paired = list(zip(combined_src, combined_tgt))
    rng.shuffle(paired)
    combined_src, combined_tgt = zip(*paired)

    logger.info(f"Total multilingual pairs: {len(combined_src):,}")
    return list(combined_src), list(combined_tgt)


def build_multilingual_hf_dataset(
    pair_configs: Dict[str, int],
    tokenizer,
    max_source_length: int = 128,
    max_target_length: int = 128,
    add_reverse: bool = True,
    temperature: float = DEFAULT_TEMPERATURE,
    val_ratio: float = 0.02,
    seed: int = 42,
):
    """
    Full pipeline: download → tag → tokenize → HF DatasetDict.
    Ready for Seq2SeqTrainer.
    """
    from datasets import Dataset, DatasetDict

    sources, targets = build_multilingual_corpus(
        pair_configs, add_reverse=add_reverse,
        temperature=temperature, seed=seed,
    )

    data = {"src": sources, "tgt": targets}
    ds = Dataset.from_dict(data)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["src"],
            max_length=max_source_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["tgt"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = ds.map(preprocess, batched=True, remove_columns=["src", "tgt"])

    n_val = max(100, int(len(ds) * val_ratio))
    split = ds.train_test_split(test_size=n_val, shuffle=True, seed=seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


# ---------------------------------------------------------------------------
# Default pair config for a full multilingual model
# ---------------------------------------------------------------------------

DEFAULT_PAIR_CONFIGS = {
    "en-vi": 500_000,
    "zh-vi": 200_000,
    "ja-vi": 200_000,
    "ko-vi": 200_000,
}
