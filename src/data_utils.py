"""
Data utilities: download OPUS parallel corpora, preprocess for MarianMT training.
Supported sources: OPUS-100, CCAligned, FLORES-200 (eval).
"""

import os
import csv
import logging
from typing import Optional, Tuple, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ---------------------------------------------------------------------------
# Corpus download
# ---------------------------------------------------------------------------

OPUS_DATASETS = {
    "en-vi": ("Helsinki-NLP/opus-100", "en-vi"),
    "zh-en": ("Helsinki-NLP/opus-100", "zh-en"),
    "ja-en": ("Helsinki-NLP/opus-100", "ja-en"),
    "ko-en": ("Helsinki-NLP/opus-100", "ko-en"),
}


def download_opus(pair_id: str, split: str = "train", max_samples: int = 200_000) -> Tuple[list, list]:
    """
    Download an OPUS-100 language pair from HuggingFace datasets.
    Returns (sources, targets).
    """
    from datasets import load_dataset

    dataset_name, hf_name = OPUS_DATASETS.get(pair_id, (None, None))
    if dataset_name is None:
        raise ValueError(f"No OPUS download config for pair: {pair_id}")

    logger.info(f"Downloading {dataset_name} {hf_name} ({split}) ...")
    ds = load_dataset(dataset_name, hf_name, split=split)
    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    src_lang, tgt_lang = pair_id.split("-")
    sources = [item["translation"][src_lang] for item in ds]
    targets = [item["translation"][tgt_lang] for item in ds]
    return sources, targets


def download_flores(lang_code: str, split: str = "devtest") -> list:
    """
    Download FLORES-200 sentences for evaluation.
    lang_code examples: 'eng_Latn', 'vie_Latn', 'zho_Hans', 'jpn_Jpan', 'kor_Hang'
    """
    from datasets import load_dataset

    flores_codes = {
        "en": "eng_Latn",
        "vi": "vie_Latn",
        "zh": "zho_Hans",
        "ja": "jpn_Jpan",
        "ko": "kor_Hang",
    }
    code = flores_codes.get(lang_code, lang_code)
    logger.info(f"Downloading FLORES-200 {code} ({split}) ...")
    ds = load_dataset("facebook/flores", code, split=split)
    return [item["sentence"] for item in ds]


# ---------------------------------------------------------------------------
# JSONL corpus loading (format: {"source": "...", "target": "..."})
# ---------------------------------------------------------------------------

def load_jsonl(path: str, max_lines: int = 0) -> Tuple[list, list]:
    """
    Load parallel corpus từ JSONL file.
    Hỗ trợ các field: source/target hoặc src/tgt hoặc translation.src/translation.tgt
    """
    import json
    sources, targets = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # Thử các format field khác nhau
            src = item.get("source") or item.get("src") or \
                  (item.get("translation", {}) or {}).get("source") or \
                  (item.get("translation", {}) or {}).get("src")
            tgt = item.get("target") or item.get("tgt") or \
                  (item.get("translation", {}) or {}).get("target") or \
                  (item.get("translation", {}) or {}).get("tgt")
            if src and tgt:
                sources.append(src.strip())
                targets.append(tgt.strip())
    logger.info(f"Loaded {len(sources):,} pairs from {path}")
    return sources, targets


def detect_format(path: str) -> str:
    """Detect file format: jsonl, tsv, or txt."""
    if path.endswith(".jsonl") or path.endswith(".json"):
        return "jsonl"
    if path.endswith(".tsv") or path.endswith(".csv"):
        return "tsv"
    # Peek at first line
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
    if first.startswith("{"):
        return "jsonl"
    if "\t" in first:
        return "tsv"
    return "txt"


def load_auto(path: str, max_lines: int = 0) -> Tuple[list, list]:
    """Auto-detect format và load file."""
    fmt = detect_format(path)
    if fmt == "jsonl":
        return load_jsonl(path, max_lines)
    elif fmt == "tsv":
        return load_tsv(path, max_lines=max_lines)
    else:
        raise ValueError(f"Unsupported format for {path}. Use .jsonl or .tsv")


# ---------------------------------------------------------------------------
# TSV / CSV corpus loading
# ---------------------------------------------------------------------------

def load_tsv(path: str, src_col: int = 0, tgt_col: int = 1, max_lines: int = 0) -> Tuple[list, list]:
    """Load a tab-separated parallel corpus file."""
    sources, targets = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if max_lines and i >= max_lines:
                break
            if len(row) > max(src_col, tgt_col):
                sources.append(row[src_col].strip())
                targets.append(row[tgt_col].strip())
    return sources, targets


def load_parallel_files(src_path: str, tgt_path: str, max_lines: int = 0) -> Tuple[list, list]:
    """Load two aligned text files (one sentence per line)."""
    sources, targets = [], []
    with open(src_path, encoding="utf-8") as sf, open(tgt_path, encoding="utf-8") as tf:
        for i, (s, t) in enumerate(zip(sf, tf)):
            if max_lines and i >= max_lines:
                break
            s, t = s.strip(), t.strip()
            if s and t:
                sources.append(s)
                targets.append(t)
    return sources, targets


def save_tsv(sources: list, targets: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for s, t in zip(sources, targets):
            writer.writerow([s, t])
    logger.info(f"Saved {len(sources)} pairs → {path}")


# ---------------------------------------------------------------------------
# HuggingFace Dataset builder for Seq2SeqTrainer
# ---------------------------------------------------------------------------

def build_hf_dataset(
    sources: list,
    targets: list,
    tokenizer,
    max_length: int = 512,
    val_ratio: float = 0.05,
):
    """
    Build a HuggingFace DatasetDict with train/validation splits,
    tokenized and ready for Seq2SeqTrainer.
    """
    from datasets import Dataset, DatasetDict

    n_val = max(1, int(len(sources) * val_ratio))
    data = {"translation": [{"src": s, "tgt": t} for s, t in zip(sources, targets)]}
    ds = Dataset.from_dict(data)

    def preprocess(batch):
        src_texts = [item["src"] for item in batch["translation"]]
        tgt_texts = [item["tgt"] for item in batch["translation"]]
        model_inputs = tokenizer(
            src_texts, max_length=max_length, truncation=True, padding=False
        )
        labels = tokenizer(
            text_target=tgt_texts, max_length=max_length, truncation=True, padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    ds = ds.map(preprocess, batched=True, remove_columns=["translation"])
    split = ds.train_test_split(test_size=n_val, shuffle=True, seed=42)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

def filter_pairs(sources: list, targets: list, max_ratio: float = 9.0) -> Tuple[list, list]:
    """Remove empty, too-long, or highly imbalanced sentence pairs."""
    kept_s, kept_t = [], []
    for s, t in zip(sources, targets):
        if not s.strip() or not t.strip():
            continue
        ratio = max(len(s), len(t)) / max(min(len(s), len(t)), 1)
        if ratio > max_ratio:
            continue
        if len(s.split()) > 200 or len(t.split()) > 200:
            continue
        kept_s.append(s)
        kept_t.append(t)
    removed = len(sources) - len(kept_s)
    if removed:
        logger.info(f"Filtered {removed}/{len(sources)} pairs ({removed/len(sources)*100:.1f}%)")
    return kept_s, kept_t
