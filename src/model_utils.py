"""
Model management utilities:
  - Download pretrained HuggingFace MarianMT models
  - Convert to CTranslate2 (INT8) for fast, memory-efficient inference
  - List available/downloaded models
"""

import os
import logging
import subprocess
import shutil
from typing import List, Optional

from .config import LANGUAGE_PAIRS, PIVOT_MODELS, MODELS_DIR

logger = logging.getLogger(__name__)


def _ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_model(pair_id: str, force: bool = False) -> str:
    """
    Download a HuggingFace MarianMT model and return its local cache path.
    Works for both direct pairs and pivot models.
    """
    from transformers import MarianMTModel, MarianTokenizer

    cfg = LANGUAGE_PAIRS.get(pair_id)
    hf_name = cfg.hf_model if cfg else PIVOT_MODELS.get(pair_id)
    if not hf_name:
        raise ValueError(f"No HuggingFace model registered for {pair_id}")

    logger.info(f"Downloading {hf_name} ...")
    tokenizer = MarianTokenizer.from_pretrained(hf_name)
    model = MarianMTModel.from_pretrained(hf_name)

    save_path = os.path.join(MODELS_DIR, f"hf_{pair_id}")
    if os.path.isdir(save_path) and not force:
        logger.info(f"Already exists at {save_path}, skipping.")
        return save_path

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Saved to {save_path}")
    return save_path


def download_all_models(include_pivot: bool = True):
    """Download all registered models."""
    _ensure_models_dir()
    for pair_id, cfg in LANGUAGE_PAIRS.items():
        if cfg.hf_model:
            try:
                download_model(pair_id)
            except Exception as e:
                logger.warning(f"Failed to download {pair_id}: {e}")

    if include_pivot:
        for pair_id in PIVOT_MODELS:
            try:
                download_model(pair_id)
            except Exception as e:
                logger.warning(f"Failed to download pivot {pair_id}: {e}")


# ---------------------------------------------------------------------------
# CTranslate2 conversion (INT8 quantization)
# ---------------------------------------------------------------------------

def convert_to_ct2(
    pair_id: str,
    quantization: str = "int8",
    source_path: Optional[str] = None,
) -> str:
    """
    Convert a MarianMT model to CTranslate2 format.

    quantization: 'int8' (smallest, fastest), 'int8_float16', 'float16', 'float32'

    Returns path to converted model directory.
    """
    try:
        import ctranslate2
    except ImportError:
        raise ImportError("Install ctranslate2: pip install ctranslate2")

    _ensure_models_dir()

    # Resolve source model path
    if source_path is None:
        ft_path = os.path.join(MODELS_DIR, f"ft_{pair_id}")
        hf_cache = os.path.join(MODELS_DIR, f"hf_{pair_id}")
        cfg = LANGUAGE_PAIRS.get(pair_id)
        hf_model_id = cfg.hf_model if cfg else PIVOT_MODELS.get(pair_id)

        if os.path.isdir(ft_path):
            source_path = ft_path
            out_suffix = "ct2_ft"
        elif os.path.isdir(hf_cache):
            source_path = hf_cache
            out_suffix = "ct2"
        elif hf_model_id:
            source_path = hf_model_id  # download on-the-fly
            out_suffix = "ct2"
        else:
            raise ValueError(f"No source model found for {pair_id}")
    else:
        out_suffix = "ct2"

    output_path = os.path.join(MODELS_DIR, f"{out_suffix}_{pair_id}")

    logger.info(f"Converting {source_path} → {output_path} ({quantization}) ...")
    ctranslate2.converters.OpusMTConverter(source_path).convert(
        output_path,
        quantization=quantization,
        force=True,
    )
    logger.info(f"Converted model saved to {output_path}")
    return output_path


def convert_all_to_ct2(quantization: str = "int8"):
    """Convert all downloaded models to CTranslate2."""
    for pair_id, cfg in LANGUAGE_PAIRS.items():
        if cfg.hf_model:
            try:
                convert_to_ct2(pair_id, quantization=quantization)
            except Exception as e:
                logger.warning(f"CT2 conversion failed for {pair_id}: {e}")
    for pair_id in PIVOT_MODELS:
        try:
            convert_to_ct2(pair_id, quantization=quantization)
        except Exception as e:
            logger.warning(f"CT2 conversion failed for pivot {pair_id}: {e}")


# ---------------------------------------------------------------------------
# Status listing
# ---------------------------------------------------------------------------

def list_models() -> List[dict]:
    """Return status of all models (downloaded, converted, fine-tuned)."""
    rows = []
    _ensure_models_dir()

    all_pairs = dict(LANGUAGE_PAIRS)
    for pid in PIVOT_MODELS:
        if pid not in all_pairs:
            all_pairs[pid] = None  # pivot only

    for pair_id, cfg in all_pairs.items():
        hf_model = cfg.hf_model if cfg else PIVOT_MODELS.get(pair_id, "—")
        hf_cache = os.path.join(MODELS_DIR, f"hf_{pair_id}")
        ct2_path = os.path.join(MODELS_DIR, f"ct2_{pair_id}")
        ft_path = os.path.join(MODELS_DIR, f"ft_{pair_id}")

        rows.append({
            "pair": pair_id,
            "hf_model": hf_model or "—",
            "downloaded": "✓" if os.path.isdir(hf_cache) else "✗",
            "ct2": "✓" if os.path.isdir(ct2_path) else "✗",
            "fine_tuned": "✓" if os.path.isdir(ft_path) else "✗",
        })
    return rows


def print_model_status():
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Model Status", show_lines=True)
    for col in ["Pair", "HF Model", "Downloaded", "CT2 (INT8)", "Fine-Tuned"]:
        table.add_column(col, style="cyan" if col == "Pair" else "white")

    for row in list_models():
        dl = "[green]✓[/green]" if row["downloaded"] == "✓" else "[red]✗[/red]"
        ct = "[green]✓[/green]" if row["ct2"] == "✓" else "[red]✗[/red]"
        ft = "[green]✓[/green]" if row["fine_tuned"] == "✓" else "[dim]—[/dim]"
        table.add_row(row["pair"], row["hf_model"], dl, ct, ft)

    console.print(table)
