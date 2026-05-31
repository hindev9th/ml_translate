"""
Train a MarianMT model from scratch (for custom language pairs or domains).

For most use-cases, finetune.py (fine-tuning a pretrained model) gives better
results with far less data. Use this script only when:
  - No pretrained model exists for your language pair
  - You have 500K+ sentence pairs
  - You need a domain-specific model trained from scratch

Architecture: standard MarianMT (6-layer encoder-decoder Transformer)
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # Data
    pair_id: str = "en-vi"
    train_tsv: Optional[str] = None
    train_src_file: Optional[str] = None
    train_tgt_file: Optional[str] = None
    use_opus: bool = True
    max_train_samples: int = 500_000
    val_ratio: float = 0.02

    # Architecture (MarianMT-small, ~77M params, fits 4GB RAM for inference)
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

    # Tokenizer
    vocab_size: int = 32_000
    spm_model: Optional[str] = None           # pre-trained SentencePiece model

    # Training
    output_dir: Optional[str] = None
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-4
    warmup_steps: int = 4_000
    max_length: int = 128
    fp16: bool = False
    label_smoothing_factor: float = 0.1
    save_total_limit: int = 3
    eval_steps: int = 2_000
    save_steps: int = 2_000
    logging_steps: int = 200
    seed: int = 42


def train_sentencepiece(texts: List[str], vocab_size: int, model_prefix: str):
    """Train a SentencePiece tokenizer on the combined corpus."""
    import sentencepiece as spm

    tmp_file = model_prefix + "_corpus.txt"
    with open(tmp_file, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.strip() + "\n")

    spm.SentencePieceTrainer.train(
        input=tmp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
        pad_id=1,
        unk_id=0,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        num_threads=4,
    )
    os.remove(tmp_file)
    logger.info(f"SentencePiece model saved: {model_prefix}.model")
    return model_prefix + ".model"


def build_marian_config(cfg: TrainConfig, tokenizer) -> dict:
    """Build MarianConfig from TrainConfig."""
    from transformers import MarianConfig

    return MarianConfig(
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


def train(cfg: TrainConfig):
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

    from .config import MODELS_DIR
    from .data_utils import (
        load_tsv, load_parallel_files, download_opus,
        build_hf_dataset, filter_pairs,
    )

    output_dir = cfg.output_dir or os.path.join(MODELS_DIR, f"trained_{cfg.pair_id}")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load data ----
    sources, targets = [], []

    if cfg.train_tsv:
        s, t = load_tsv(cfg.train_tsv)
        sources.extend(s); targets.extend(t)

    if cfg.train_src_file and cfg.train_tgt_file:
        s, t = load_parallel_files(cfg.train_src_file, cfg.train_tgt_file)
        sources.extend(s); targets.extend(t)

    if cfg.use_opus or not sources:
        src, tgt = cfg.pair_id.split("-")
        opus_pair = f"{src}-{tgt}" if tgt != "vi" else f"{src}-en"
        s, t = download_opus(opus_pair, max_samples=cfg.max_train_samples)
        sources.extend(s); targets.extend(t)

    sources, targets = filter_pairs(sources, targets)
    sources = sources[: cfg.max_train_samples]
    targets = targets[: cfg.max_train_samples]
    logger.info(f"Training pairs: {len(sources):,}")

    # ---- Tokenizer ----
    if cfg.spm_model and os.path.isfile(cfg.spm_model):
        spm_path = cfg.spm_model
    else:
        spm_prefix = os.path.join(output_dir, "spm_model")
        logger.info("Training SentencePiece tokenizer ...")
        spm_path = train_sentencepiece(
            sources + targets, cfg.vocab_size, spm_prefix
        )

    tokenizer = MarianTokenizer(vocab_file=spm_path)
    tokenizer.save_pretrained(output_dir)

    # ---- Model ----
    model_cfg = build_marian_config(cfg, tokenizer)
    model = MarianMTModel(model_cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {n_params:.1f}M")

    # ---- Dataset ----
    dataset = build_hf_dataset(
        sources, targets, tokenizer,
        max_length=cfg.max_length,
        val_ratio=cfg.val_ratio,
    )

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

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=cfg.max_length,
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
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model,
        label_pad_token_id=-100, pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    logger.info("Starting training from scratch ...")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"Model saved to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train MarianMT from scratch")
    parser.add_argument("--pair", default="en-vi")
    parser.add_argument("--train-tsv", default=None)
    parser.add_argument("--train-src", default=None)
    parser.add_argument("--train-tgt", default=None)
    parser.add_argument("--use-opus", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = TrainConfig(
        pair_id=args.pair,
        train_tsv=args.train_tsv,
        train_src_file=args.train_src,
        train_tgt_file=args.train_tgt,
        use_opus=args.use_opus,
        max_train_samples=args.max_samples,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        fp16=args.fp16,
        output_dir=args.output_dir,
    )
    out = train(cfg)
    print(f"\nTrained model saved to: {out}")
