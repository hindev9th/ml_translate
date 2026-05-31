"""
Fine-tuning a pretrained MarianMT model on custom parallel data.

Features:
  - Memory-efficient: gradient checkpointing, fp16, small batches
  - Runs on 4 GB RAM (CPU) or GPU if available
  - Evaluates BLEU every N steps, saves best checkpoint
  - Supports TSV, parallel files, and HuggingFace OPUS datasets
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FinetuneConfig:
    # Model
    pair_id: str = "en-vi"                          # e.g. 'en-vi', 'zh-vi'
    base_model: Optional[str] = None                # overrides LANGUAGE_PAIRS[pair_id].hf_model

    # Data
    train_tsv: Optional[str] = None                 # TSV file: src\ttgt
    train_src_file: Optional[str] = None            # parallel src file
    train_tgt_file: Optional[str] = None            # parallel tgt file
    use_opus: bool = False                           # download OPUS-100 corpus
    max_train_samples: int = 100_000
    max_eval_samples: int = 2_000
    val_ratio: float = 0.05

    # Training
    output_dir: Optional[str] = None                # defaults to models/ft_{pair_id}
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    max_source_length: int = 128
    max_target_length: int = 128
    fp16: bool = False                               # set True if GPU with fp16 support
    gradient_checkpointing: bool = True              # saves ~30% RAM during training
    save_total_limit: int = 2
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    metric_for_best_model: str = "eval_bleu"
    greater_is_better: bool = True
    seed: int = 42


def finetune(cfg: FinetuneConfig):
    import torch
    from transformers import (
        MarianMTModel,
        MarianTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
    )
    import evaluate as hf_evaluate

    from .config import LANGUAGE_PAIRS, MODELS_DIR
    from .data_utils import (
        load_tsv,
        load_parallel_files,
        download_opus,
        build_hf_dataset,
        filter_pairs,
    )

    # ---- Resolve model ID ----
    pair_cfg = LANGUAGE_PAIRS.get(cfg.pair_id)
    base_model = cfg.base_model or (pair_cfg.hf_model if pair_cfg else None)
    if not base_model:
        raise ValueError(
            f"No base model for {cfg.pair_id}. "
            "Pass --base-model Helsinki-NLP/opus-mt-... explicitly."
        )

    output_dir = cfg.output_dir or os.path.join(MODELS_DIR, f"ft_{cfg.pair_id}")
    os.makedirs(output_dir, exist_ok=True)

    # TensorBoard log dir (transformers 5.x dùng env var)
    tb_dir = os.path.join(output_dir, "runs")
    os.makedirs(tb_dir, exist_ok=True)
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", tb_dir)

    # ---- Load tokenizer & model ----
    logger.info(f"Loading base model: {base_model}")
    tokenizer = MarianTokenizer.from_pretrained(base_model)
    model = MarianMTModel.from_pretrained(base_model)

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ---- Load data ----
    sources, targets = [], []

    if cfg.train_tsv:
        logger.info(f"Loading data from {cfg.train_tsv}")
        s, t = load_tsv(cfg.train_tsv)
        sources.extend(s); targets.extend(t)

    if cfg.train_src_file and cfg.train_tgt_file:
        logger.info(f"Loading parallel files: {cfg.train_src_file}, {cfg.train_tgt_file}")
        s, t = load_parallel_files(cfg.train_src_file, cfg.train_tgt_file)
        sources.extend(s); targets.extend(t)

    if cfg.use_opus:
        opus_pair = cfg.pair_id
        # For pivot pairs, download the constituent pairs
        src, tgt = cfg.pair_id.split("-")
        if tgt == "vi" and src != "en":
            # Download src-en corpus for this pair
            opus_pair = f"{src}-en"
        logger.info(f"Downloading OPUS-100 corpus for {opus_pair} ...")
        s, t = download_opus(opus_pair, max_samples=cfg.max_train_samples)
        sources.extend(s); targets.extend(t)

    if not sources:
        raise ValueError(
            "No training data! Provide --train-tsv, --train-src/tgt, or --use-opus"
        )

    sources, targets = filter_pairs(sources, targets)
    if cfg.max_train_samples:
        sources = sources[: cfg.max_train_samples]
        targets = targets[: cfg.max_train_samples]

    logger.info(f"Total training pairs: {len(sources):,}")

    # ---- Build HF dataset ----
    dataset = build_hf_dataset(
        sources, targets, tokenizer,
        max_length=cfg.max_source_length,
        val_ratio=cfg.val_ratio,
    )

    # ---- BLEU metric ----
    sacrebleu_metric = hf_evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        import numpy as np

        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = sacrebleu_metric.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
        )
        return {"bleu": round(result["score"], 2)}

    # ---- Optimizer (8-bit Adam nếu bitsandbytes được cài) ----
    use_bnb = False
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa
            use_bnb = True
            logger.info("bitsandbytes found — using 8-bit AdamW (saves ~50% VRAM)")
        except ImportError:
            pass

    # ---- Training args ----
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
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
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        seed=cfg.seed,
        report_to="tensorboard",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
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
            EarlyStoppingCallback(early_stopping_patience=3),
            TranslationSampleCallback(tokenizer, beam_size=4),
        ],
    )

    logger.info("Starting fine-tuning ...")
    trainer.train()

    logger.info(f"Saving best model → {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Fine-tuning complete.")

    return output_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Fine-tune MarianMT model")
    parser.add_argument("--pair", default="en-vi", help="Language pair, e.g. en-vi")
    parser.add_argument("--base-model", default=None, help="Override base HF model ID")
    parser.add_argument("--train-tsv", default=None, help="TSV file with src\\ttgt")
    parser.add_argument("--train-src", default=None, help="Source sentences file")
    parser.add_argument("--train-tgt", default=None, help="Target sentences file")
    parser.add_argument("--use-opus", action="store_true", help="Use OPUS-100 corpus")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = FinetuneConfig(
        pair_id=args.pair,
        base_model=args.base_model,
        train_tsv=args.train_tsv,
        train_src_file=args.train_src,
        train_tgt_file=args.train_tgt,
        use_opus=args.use_opus,
        max_train_samples=args.max_samples,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.fp16,
        output_dir=args.output_dir,
    )
    out = finetune(cfg)
    print(f"\nFine-tuned model saved to: {out}")
