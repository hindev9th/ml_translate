"""
Evaluation utilities: BLEU, chrF, TER scores.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def compute_bleu(hypotheses: List[str], references: List[str]) -> dict:
    """Compute sacreBLEU score."""
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        "bleu": round(bleu.score, 2),
        "bleu_signature": bleu.format(signature=True),
    }


def compute_chrf(hypotheses: List[str], references: List[str]) -> dict:
    import sacrebleu

    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    return {"chrf": round(chrf.score, 2)}


def compute_ter(hypotheses: List[str], references: List[str]) -> dict:
    import sacrebleu

    ter = sacrebleu.corpus_ter(hypotheses, [references])
    return {"ter": round(ter.score, 2)}


def evaluate_model(
    engine,
    src_texts: List[str],
    ref_texts: List[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 32,
) -> dict:
    """
    Translate src_texts and evaluate against ref_texts.
    Returns dict with bleu, chrf, ter scores.
    """
    logger.info(f"Evaluating {src_lang}→{tgt_lang} on {len(src_texts)} sentences ...")
    hypotheses = []
    for i in range(0, len(src_texts), batch_size):
        batch = src_texts[i : i + batch_size]
        hyps = engine.translate_batch(batch, src=src_lang, tgt=tgt_lang)
        hypotheses.extend(hyps)

    results = {}
    results.update(compute_bleu(hypotheses, ref_texts))
    results.update(compute_chrf(hypotheses, ref_texts))
    results.update(compute_ter(hypotheses, ref_texts))
    results["n_sentences"] = len(src_texts)

    logger.info(
        f"Results → BLEU: {results['bleu']}, chrF: {results['chrf']}, TER: {results['ter']}"
    )
    return results, hypotheses


def evaluate_flores(engine, src_lang: str, tgt_lang: str) -> dict:
    """Evaluate on FLORES-200 devtest set (easy standard benchmark)."""
    from .data_utils import download_flores

    src_texts = download_flores(src_lang)
    ref_texts = download_flores(tgt_lang)
    n = min(len(src_texts), len(ref_texts))
    results, _ = evaluate_model(engine, src_texts[:n], ref_texts[:n], src_lang, tgt_lang)
    return results
