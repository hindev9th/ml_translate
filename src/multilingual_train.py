"""
Train a single multilingual MarianMT model from scratch.

One model handles all 8 directions:
  EN↔VI, ZH↔VI, JA↔VI, KO↔VI

How it works:
  - Shared SentencePiece BPE tokenizer (50K vocab, all 5 languages)
  - Language prefix tokens on the source: >>vi<< Hello → Xin chào
  - Mixed batches from all language pairs simultaneously
  - Temperature sampling keeps balance between large/small corpora

Minimum recommended data:
  EN-VI: 200K+, ZH-VI: 100K+, JA-VI: 100K+, KO-VI: 100K+

Usage:
    python src/multilingual_train.py
    python src/multilingual_train.py --config configs/multilingual.yaml
    python src/multilingual_train.py --epochs 10 --batch-size 16 --vocab-size 50000
    python src/multilingual_train.py --resume models/custom_vi/checkpoint-5000
"""

import os
import sys
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.dirname(__file__))
DEFAULT_OUTPUT = os.path.join(_HERE, "models", "custom_vi")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MultilingualTrainConfig:
    # Output
    output_dir: str = DEFAULT_OUTPUT

    # Data — max samples per pair (both directions will be included)
    pair_configs: Dict[str, int] = field(default_factory=lambda: {
        "en-vi": 500_000,
        "zh-vi": 200_000,
        "ja-vi": 200_000,
        "ko-vi": 200_000,
    })
    # Local TSV files override auto-download (key = pair_id, value = path)
    local_files: Dict[str, Optional[str]] = field(default_factory=dict)
    add_reverse: bool = True          # also train VI→ZH, VI→JA, etc.
    temperature: float = 0.7          # sampling temperature for balance
    val_ratio: float = 0.02

    # Warm start — khuyến nghị dùng thay vì random init
    # Dùng weights từ pretrained model → hội tụ nhanh hơn 5-10x
    warm_start_model: Optional[str] = "Helsinki-NLP/opus-mt-en-vi"

    # Tokenizer
    vocab_size: int = 50_000
    spm_model: Optional[str] = None   # pre-trained SPM, or train fresh

    # Architecture (fits comfortably in 4GB RAM at inference, INT8)
    encoder_layers: int = 6
    decoder_layers: int = 6
    d_model: int = 512
    encoder_ffn_dim: int = 2048
    decoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 8
    decoder_attention_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_position_embeddings: int = 512
    label_smoothing_factor: float = 0.1

    # Training
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2     # effective batch = 32
    learning_rate: float = 1e-4              # 1e-4 ổn định hơn 5e-4 khi train từ đầu
    warmup_steps: int = 8_000               # warmup dài hơn để LR tăng chậm
    max_grad_norm: float = 0.5              # strict clipping chống gradient explosion
    max_source_length: int = 128
    max_target_length: int = 128
    fp16: bool = False
    gradient_checkpointing: bool = True
    save_total_limit: int = 3
    eval_steps: int = 2_000
    save_steps: int = 2_000
    logging_steps: int = 100
    early_stopping_patience: int = 5
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    # Giới hạn eval samples để tăng tốc (0 = dùng tất cả)
    max_eval_samples: int = 2000

    @classmethod
    def from_yaml(cls, path: str) -> "MultilingualTrainConfig":
        """Load config from a YAML file (configs/multilingual.yaml)."""
        import yaml

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        cfg = cls()
        cfg.output_dir = raw.get("output_dir", cfg.output_dir)
        cfg.warm_start_model = raw.get("warm_start_model", cfg.warm_start_model)

        d = raw.get("data", {})
        if "pair_configs" in d:
            cfg.pair_configs = d["pair_configs"]
        lf = d.get("local_files", {})
        cfg.local_files = {k: v for k, v in lf.items() if v}  # drop nulls
        cfg.add_reverse  = d.get("add_reverse",  cfg.add_reverse)
        cfg.temperature  = d.get("temperature",  cfg.temperature)
        cfg.val_ratio    = d.get("val_ratio",     cfg.val_ratio)
        cfg.seed         = d.get("seed",          cfg.seed)

        tok = raw.get("tokenizer", {})
        cfg.vocab_size = tok.get("vocab_size", cfg.vocab_size)
        if tok.get("spm_model"):
            cfg.spm_model = tok["spm_model"]

        m = raw.get("model", {})
        cfg.encoder_layers          = m.get("encoder_layers",          cfg.encoder_layers)
        cfg.decoder_layers          = m.get("decoder_layers",          cfg.decoder_layers)
        cfg.d_model                 = m.get("d_model",                 cfg.d_model)
        cfg.encoder_ffn_dim         = m.get("encoder_ffn_dim",         cfg.encoder_ffn_dim)
        cfg.decoder_ffn_dim         = m.get("decoder_ffn_dim",         cfg.decoder_ffn_dim)
        cfg.encoder_attention_heads = m.get("encoder_attention_heads", cfg.encoder_attention_heads)
        cfg.decoder_attention_heads = m.get("decoder_attention_heads", cfg.decoder_attention_heads)
        cfg.dropout                 = m.get("dropout",                 cfg.dropout)
        cfg.label_smoothing_factor  = m.get("label_smoothing_factor",  cfg.label_smoothing_factor)

        t = raw.get("training", {})
        cfg.num_train_epochs              = t.get("num_train_epochs",              cfg.num_train_epochs)
        cfg.per_device_train_batch_size   = t.get("per_device_train_batch_size",   cfg.per_device_train_batch_size)
        cfg.per_device_eval_batch_size    = t.get("per_device_eval_batch_size",    cfg.per_device_eval_batch_size)
        cfg.gradient_accumulation_steps   = t.get("gradient_accumulation_steps",   cfg.gradient_accumulation_steps)
        cfg.learning_rate                 = t.get("learning_rate",                 cfg.learning_rate)
        cfg.warmup_steps                  = t.get("warmup_steps",                  cfg.warmup_steps)
        cfg.max_grad_norm                 = t.get("max_grad_norm",                 cfg.max_grad_norm)
        cfg.max_source_length             = t.get("max_source_length",             cfg.max_source_length)
        cfg.max_target_length             = t.get("max_target_length",             cfg.max_target_length)
        cfg.fp16                          = t.get("fp16",                          cfg.fp16)
        cfg.gradient_checkpointing        = t.get("gradient_checkpointing",        cfg.gradient_checkpointing)
        cfg.save_total_limit              = t.get("save_total_limit",              cfg.save_total_limit)
        cfg.eval_steps                    = t.get("eval_steps",                    cfg.eval_steps)
        cfg.save_steps                    = t.get("save_steps",                    cfg.save_steps)
        cfg.logging_steps                 = t.get("logging_steps",                 cfg.logging_steps)
        cfg.early_stopping_patience       = t.get("early_stopping_patience",       cfg.early_stopping_patience)
        if t.get("resume_from_checkpoint"):
            cfg.resume_from_checkpoint = t["resume_from_checkpoint"]

        return cfg


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def _train_or_load_tokenizer(cfg: MultilingualTrainConfig, corpus_texts):
    from .multilingual_tokenizer import (
        train_tokenizer, load_tokenizer, DEFAULT_SPM_MODEL
    )

    spm_path = cfg.spm_model
    if spm_path and os.path.isfile(spm_path):
        logger.info(f"Using existing SPM model: {spm_path}")
    else:
        os.makedirs(cfg.output_dir, exist_ok=True)
        spm_prefix = os.path.join(cfg.output_dir, "spm")
        logger.info("Training shared SentencePiece tokenizer ...")
        spm_path = train_tokenizer(
            corpus_texts,
            model_prefix=spm_prefix,
            vocab_size=cfg.vocab_size,
        )

    tokenizer = load_tokenizer(spm_path)
    tokenizer.save_pretrained(cfg.output_dir)
    return tokenizer


# ---------------------------------------------------------------------------
# Warm start helpers
# ---------------------------------------------------------------------------

LANG_TOKENS = [">>en<<", ">>vi<<", ">>zh<<", ">>ja<<", ">>ko<<"]


def _warm_start_tokenizer_and_model(base_model_id: str, output_dir: str):
    """
    Load pretrained MarianTokenizer + MarianMTModel.
    Add missing language tokens (>>zh<<, >>ja<<, >>ko<<) to vocab.
    Returns (tokenizer, model) ready for multilingual training.
    """
    import torch
    from transformers import MarianTokenizer, MarianMTModel

    logger.info(f"Warm start: loading {base_model_id}")
    tokenizer = MarianTokenizer.from_pretrained(base_model_id)
    model     = MarianMTModel.from_pretrained(base_model_id)

    # Add missing language tokens
    missing = [t for t in LANG_TOKENS if t not in tokenizer.get_vocab()]
    if missing:
        logger.info(f"Adding {len(missing)} language tokens: {missing}")
        tokenizer.add_tokens(missing)

        # Resize embeddings — init new rows from mean of existing embeddings
        old_emb = model.model.shared.weight.data
        model.resize_token_embeddings(len(tokenizer))
        new_emb = model.model.shared.weight.data

        n_new = len(missing)
        mean_vec = old_emb.mean(dim=0)
        for i in range(1, n_new + 1):
            new_emb[-i] = mean_vec + 0.02 * torch.randn_like(mean_vec)

        logger.info(f"Embedding matrix resized: {old_emb.shape[0]} → {len(tokenizer)}")

    tokenizer.save_pretrained(output_dir)
    return tokenizer, model


# ---------------------------------------------------------------------------
# Model construction (from scratch — used when warm_start_model is None)
# ---------------------------------------------------------------------------

def _build_model(cfg: MultilingualTrainConfig, tokenizer):
    from transformers import MarianConfig, MarianMTModel

    marian_cfg = MarianConfig(
        vocab_size=len(tokenizer),
        encoder_layers=cfg.encoder_layers,
        decoder_layers=cfg.decoder_layers,
        d_model=cfg.d_model,
        encoder_ffn_dim=cfg.encoder_ffn_dim,
        decoder_ffn_dim=cfg.decoder_ffn_dim,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_attention_heads=cfg.decoder_attention_heads,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        max_position_embeddings=cfg.max_position_embeddings,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
    )

    model = MarianMTModel(marian_cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model: {n_params:.1f}M parameters")
    return model


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: MultilingualTrainConfig = None):
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )
    import evaluate as hf_evaluate

    from .multilingual_data import (
        build_multilingual_corpus,
        build_multilingual_hf_dataset,
    )

    if cfg is None:
        cfg = MultilingualTrainConfig()

    os.makedirs(cfg.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    logger.info("=" * 60)
    logger.info("  Multilingual MarianMT Training")
    logger.info(f"  Pairs: {list(cfg.pair_configs.keys())}")
    logger.info(f"  Output: {cfg.output_dir}")
    logger.info("=" * 60)

    # ---- Step 1: Collect file paths (KHÔNG load vào RAM ngay) ----
    logger.info("Step 1/4: Resolving data sources ...")
    from .multilingual_data import download_pair, prepare_pairs
    from .data_utils import filter_pairs

    # Chỉ lưu (pair_id, file_path, max_n) — chưa đọc dữ liệu
    data_sources: list = []   # [(pair_id, src_lang, tgt_lang, local_path, max_n)]

    for pair_id, max_n in cfg.pair_configs.items():
        src_lang, tgt_lang = pair_id.split("-")
        local_path = cfg.local_files.get(pair_id)
        if local_path and os.path.isfile(local_path):
            logger.info(f"  {pair_id}: {local_path} (max={max_n:,})")
            data_sources.append((pair_id, src_lang, tgt_lang, local_path, max_n))
        else:
            logger.info(f"  {pair_id}: will download (max={max_n:,})")
            data_sources.append((pair_id, src_lang, tgt_lang, None, max_n))

    if not data_sources:
        raise RuntimeError("No data sources found.")

    # ---- Step 2: Tokenizer + Model (trước khi load data) ----
    if cfg.warm_start_model:
        logger.info(f"Step 2/4: Warm start from {cfg.warm_start_model} ...")
        tokenizer, pretrained_model = _warm_start_tokenizer_and_model(
            cfg.warm_start_model, cfg.output_dir
        )
    else:
        # Cần sample nhỏ để train tokenizer — chỉ load 50K/pair
        logger.info("Step 2/4: Training shared tokenizer (sampling corpus) ...")
        sample_corpus = []
        for pair_id, src_lang, tgt_lang, local_path, max_n in data_sources:
            if local_path:
                from .data_utils import load_auto
                s, t = load_auto(local_path, max_lines=50_000)
                sample_corpus.extend(s); sample_corpus.extend(t)
        tokenizer = _train_or_load_tokenizer(cfg, sample_corpus)
        del sample_corpus
        pretrained_model = None
    logger.info(f"Vocabulary size: {len(tokenizer):,}")

    # ---- Step 3: Build HF dataset (với disk cache để tránh map lại) ----
    logger.info("Step 3/4: Tokenizing and building dataset ...")

    # Cache key dựa trên config — thay đổi data/tokenizer = cache mới
    import hashlib, json as _json
    _cache_key = hashlib.md5(_json.dumps({
        "local_files": sorted(cfg.local_files.items()),
        "pair_configs": sorted(cfg.pair_configs.items()),
        "add_reverse": cfg.add_reverse,
        "max_src": cfg.max_source_length,
        "max_tgt": cfg.max_target_length,
        "warm_start": cfg.warm_start_model,
        "seed": cfg.seed,
    }, sort_keys=True).encode()).hexdigest()[:10]

    cache_dir = os.path.join(cfg.output_dir, f"dataset_cache_{_cache_key}")

    from datasets import Dataset, DatasetDict

    _cache_ready = os.path.join(cache_dir, "_CACHE_COMPLETE")

    if os.path.isdir(cache_dir) and os.path.isfile(_cache_ready):
        logger.info(f"Loading tokenized dataset from cache: {cache_dir}")
        logger.info("(Bỏ qua bước map — đã được cache từ lần chạy trước)")
        dataset = DatasetDict.load_from_disk(cache_dir)
        # Giới hạn eval size nếu cần
        val_ds = dataset["validation"]
        if cfg.max_eval_samples and len(val_ds) > cfg.max_eval_samples:
            val_ds = val_ds.select(range(cfg.max_eval_samples))
            dataset = DatasetDict({"train": dataset["train"], "validation": val_ds})
        logger.info(f"Train: {len(dataset['train']):,}  Val: {len(dataset['validation']):,}")
    else:
        # Dùng generator để tránh load toàn bộ 36M samples vào RAM cùng lúc
        import random

        def _pair_generator():
            """Yield từng (src, tgt) pair từ file, không giữ toàn bộ trong RAM."""
            rng_gen = random.Random(cfg.seed)
            all_items = []
            for pair_id, src_lang, tgt_lang, local_path, max_n in data_sources:
                if local_path:
                    from .data_utils import load_auto
                    srcs, tgts = load_auto(local_path, max_lines=max_n)
                else:
                    srcs, tgts = download_pair(src_lang, tgt_lang, max_samples=max_n)
                srcs, tgts = filter_pairs(srcs, tgts)
                tagged_s, tagged_t = prepare_pairs(srcs, tgts, src_lang, tgt_lang, cfg.add_reverse)
                # Giải phóng bộ nhớ ngay
                del srcs, tgts
                all_items.extend(zip(tagged_s, tagged_t))
                del tagged_s, tagged_t
                logger.info(f"  Loaded {pair_id}: {len(all_items):,} total pairs so far")

            logger.info(f"Shuffling {len(all_items):,} pairs ...")
            rng_gen.shuffle(all_items)
            for s, t in all_items:
                yield {"src": s, "tgt": t}

        logger.info("Building dataset from generator (tiết kiệm RAM) ...")
        ds = Dataset.from_generator(_pair_generator)
        total = len(ds)
        logger.info(f"Tokenizing {total:,} pairs — lan dau mat thoi gian, lan sau load tu cache ...")

        def preprocess(batch):
            model_inputs = tokenizer(
                batch["src"],
                max_length=cfg.max_source_length,
                truncation=True,
                padding=False,
            )
            labels = tokenizer(
                text_target=batch["tgt"],
                max_length=cfg.max_target_length,
                truncation=True,
                padding=False,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        ds = ds.map(preprocess, batched=True, batch_size=2000,
                    remove_columns=["src", "tgt"],
                    desc="Tokenizing")

        n_val = max(500, int(len(ds) * cfg.val_ratio))
        split = ds.train_test_split(test_size=n_val, shuffle=True, seed=cfg.seed)
        val_ds = split["test"]

        # Giới hạn eval size để eval không mất nhiều thời gian
        if cfg.max_eval_samples and len(val_ds) > cfg.max_eval_samples:
            val_ds = val_ds.select(range(cfg.max_eval_samples))
            logger.info(f"Eval limited to {cfg.max_eval_samples} samples (from {n_val:,})")

        dataset = DatasetDict({"train": split["train"], "validation": val_ds})
        logger.info(f"Train: {len(dataset['train']):,}  Val: {len(dataset['validation']):,}")

        # Lưu cache xuống disk để lần sau load nhanh
        # Kiểm tra không có process khác đang build cache
        _lock = cache_dir + ".lock"
        if os.path.exists(_lock):
            logger.warning(f"Cache lock detected at {_lock}. Another process may be building cache. Skipping save.")
        else:
            open(_lock, "w").close()
            try:
                logger.info(f"Saving tokenized dataset to cache: {cache_dir}")
                full_val = split["test"]
                DatasetDict({"train": split["train"], "validation": full_val}).save_to_disk(cache_dir)
                # Đánh dấu cache hoàn chỉnh
                open(_cache_ready, "w").close()
                logger.info("Cache saved. Lan sau se load ngay lap tuc.")
            finally:
                if os.path.exists(_lock):
                    os.remove(_lock)

    # ---- Step 4: Build model and train ----
    logger.info("Step 4/4: Building model and starting training ...")

    # Set TensorBoard log dir via env var (transformers 5.x)
    tb_dir = os.path.join(cfg.output_dir, "runs")
    os.makedirs(tb_dir, exist_ok=True)
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", tb_dir)
    logger.info(f"TensorBoard logs: {tb_dir}")

    # Dùng pretrained weights nếu có (warm start), ngược lại build từ đầu
    if pretrained_model is not None:
        model = pretrained_model
        logger.info("Using pretrained weights (warm start)")
    else:
        model = _build_model(cfg, tokenizer)
        logger.info("Using random initialization (from scratch)")

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 8-bit AdamW nếu bitsandbytes được cài (giảm ~50% VRAM)
    use_bnb = False
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa
            use_bnb = True
            logger.info("bitsandbytes found — using 8-bit AdamW optimizer")
        except ImportError:
            pass

    sacrebleu = hf_evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        import numpy as np
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds   = np.where(preds   != -100, preds,   tokenizer.pad_token_id)
        labels  = np.where(labels  != -100, labels,  tokenizer.pad_token_id)
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = sacrebleu.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
        )
        return {"bleu": round(result["score"], 2)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        optim="adamw_bnb_8bit" if use_bnb else "adamw_torch",
        predict_with_generate=True,
        generation_max_length=cfg.max_target_length,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_bleu",
        greater_is_better=True,
        label_smoothing_factor=cfg.label_smoothing_factor,
        seed=cfg.seed,
        report_to="tensorboard",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    from .callbacks import TranslationSampleCallback

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience),
            TranslationSampleCallback(tokenizer, beam_size=4),
        ],
    )

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    logger.info(f"Saving final model → {cfg.output_dir}")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("Training complete.")
    return cfg.output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train custom multilingual MarianMT (EN/ZH/JA/KO ↔ VI)"
    )
    p.add_argument("--config", default=None, metavar="YAML",
                   help="Path to YAML config (e.g. configs/multilingual.yaml). "
                        "CLI flags override YAML values.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=4_000)
    p.add_argument("--vocab-size", type=int, default=50_000)
    p.add_argument("--spm-model", default=None, help="Re-use existing SPM model path")
    p.add_argument("--en-vi",  type=int, default=500_000, metavar="N")
    p.add_argument("--zh-vi",  type=int, default=200_000, metavar="N")
    p.add_argument("--ja-vi",  type=int, default=200_000, metavar="N")
    p.add_argument("--ko-vi",  type=int, default=200_000, metavar="N")
    p.add_argument("--no-reverse", action="store_true",
                   help="Only train SRC→VI, skip VI→SRC directions")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (higher=more uniform across pairs)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--resume", default=None, help="Resume from checkpoint path")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", type=int, default=6, help="Encoder & decoder layers")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    # Start from YAML if provided, then override with CLI flags
    if args.config:
        cfg = MultilingualTrainConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        cfg = MultilingualTrainConfig()

    # CLI flags override YAML (only when explicitly provided)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.en_vi != 500_000 or args.zh_vi != 200_000:   # non-default = user set
        cfg.pair_configs = {
            "en-vi": args.en_vi, "zh-vi": args.zh_vi,
            "ja-vi": args.ja_vi, "ko-vi": args.ko_vi,
        }
    if args.no_reverse:
        cfg.add_reverse = False
    if args.temperature != 0.7:
        cfg.temperature = args.temperature
    if args.vocab_size != 50_000:
        cfg.vocab_size = args.vocab_size
    if args.spm_model:
        cfg.spm_model = args.spm_model
    if args.layers != 6:
        cfg.encoder_layers = cfg.decoder_layers = args.layers
    if args.d_model != 512:
        cfg.d_model = args.d_model
    if args.epochs != 10:
        cfg.num_train_epochs = args.epochs
    if args.batch_size != 16:
        cfg.per_device_train_batch_size = args.batch_size
    if args.accum_steps != 2:
        cfg.gradient_accumulation_steps = args.accum_steps
    if args.lr != 5e-4:
        cfg.learning_rate = args.lr
    if args.warmup != 4_000:
        cfg.warmup_steps = args.warmup
    if args.fp16:
        cfg.fp16 = True
    if args.no_gradient_checkpointing:
        cfg.gradient_checkpointing = False
    if args.resume:
        cfg.resume_from_checkpoint = args.resume

    out = train(cfg)
    print(f"\nModel saved to: {out}")
