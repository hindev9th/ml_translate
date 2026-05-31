"""
CLI interface for the Vietnamese translation system.

Commands:
  translate   — translate text or file
  interactive — interactive REPL
  server      — launch Gradio web server
  models      — list / download / convert models
  evaluate    — run BLEU evaluation
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()
logger = logging.getLogger(__name__)


def get_engine(device: str = "cpu"):
    from src.translate_engine import TranslationEngine
    return TranslationEngine(device=device, max_loaded_models=3)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug):
    """ViTrans — Máy dịch Anh/Trung/Hàn/Nhật ↔ Tiếng Việt (MarianNMT)"""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


# ============================================================
# translate
# ============================================================

@cli.command()
@click.option("--src", "-s", default="auto",
              type=click.Choice(["auto", "en", "vi", "zh", "ja", "ko"]),
              help="Source language (auto = detect automatically)")
@click.option("--tgt", "-t", required=True,
              type=click.Choice(["en", "vi", "zh", "ja", "ko"]),
              help="Target language")
@click.option("--text", "-T", default=None, help="Text to translate")
@click.option("--file", "-f", "input_file", default=None,
              type=click.Path(exists=True), help="Input text file")
@click.option("--output", "-o", default=None,
              type=click.Path(), help="Output file (default: stdout)")
@click.option("--beam", default=4, show_default=True, help="Beam size")
@click.option("--batch-size", default=32, show_default=True, help="Batch size for file mode")
@click.option("--device", default="cpu", show_default=True, help="cpu or cuda")
def translate(src, tgt, text, input_file, output, beam, batch_size, device):
    """Translate text or a file."""
    if src != "auto" and src == tgt:
        console.print("[red]Error:[/red] Source and target languages must differ.")
        sys.exit(1)

    engine = get_engine(device)

    if text:
        result = engine.translate(text, src=src, tgt=tgt, beam_size=beam)
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(result + "\n")
        else:
            console.print(Panel(result, title=f"{src.upper()} → {tgt.upper()}", border_style="green"))

    elif input_file:
        with open(input_file, encoding="utf-8") as f:
            lines = f.read().splitlines()

        non_empty = [(i, l) for i, l in enumerate(lines) if l.strip()]
        result_lines = list(lines)

        console.print(f"Translating {len(non_empty)} lines ({src}→{tgt}) ...")
        from tqdm import tqdm

        for i in tqdm(range(0, len(non_empty), batch_size)):
            batch_idx = [idx for idx, _ in non_empty[i : i + batch_size]]
            batch_texts = [txt for _, txt in non_empty[i : i + batch_size]]
            translations = engine.translate_batch(batch_texts, src=src, tgt=tgt, beam_size=beam)
            for idx, trans in zip(batch_idx, translations):
                result_lines[idx] = trans

        out_text = "\n".join(result_lines)
        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(out_text)
            console.print(f"[green]✓[/green] Saved to {output}")
        else:
            console.print(out_text)

    else:
        # Read from stdin
        console.print(f"[dim]Reading from stdin ({src}→{tgt}), Ctrl+D to finish...[/dim]")
        stdin_text = sys.stdin.read().strip()
        if stdin_text:
            result = engine.translate(stdin_text, src=src, tgt=tgt, beam_size=beam)
            console.print(result)


# ============================================================
# interactive
# ============================================================

@cli.command()
@click.option("--src", "-s", default="en",
              type=click.Choice(["en", "vi", "zh", "ja", "ko"]))
@click.option("--tgt", "-t", default="vi",
              type=click.Choice(["en", "vi", "zh", "ja", "ko"]))
@click.option("--beam", default=4)
@click.option("--device", default="cpu")
def interactive(src, tgt, beam, device):
    """Interactive translation REPL."""
    from src.config import LANG_NAMES

    engine = get_engine(device)

    console.print(Panel(
        f"[bold green]ViTrans Interactive[/bold green]\n"
        f"[cyan]{LANG_NAMES[src]}[/cyan] → [cyan]{LANG_NAMES[tgt]}[/cyan]\n\n"
        "Commands: [yellow]:swap[/yellow] [yellow]:src <code>[/yellow] [yellow]:tgt <code>[/yellow] [yellow]:quit[/yellow]",
        border_style="green",
    ))

    while True:
        try:
            text = console.input(f"[bold blue]({src}→{tgt})[/bold blue] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if not text:
            continue
        if text == ":quit":
            break
        if text == ":swap":
            src, tgt = tgt, src
            console.print(f"[dim]Swapped: {src} ↔ {tgt}[/dim]")
            continue
        if text.startswith(":src "):
            src = text.split()[1]
            console.print(f"[dim]Source: {src}[/dim]")
            continue
        if text.startswith(":tgt "):
            tgt = text.split()[1]
            console.print(f"[dim]Target: {tgt}[/dim]")
            continue

        try:
            result = engine.translate(text, src=src, tgt=tgt, beam_size=beam)
            console.print(f"[green]→[/green] {result}\n")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}\n")


# ============================================================
# server
# ============================================================

@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=7860, show_default=True)
@click.option("--share", is_flag=True, help="Create public Gradio link")
@click.option("--warmup", is_flag=True, help="Pre-load EN↔VI models")
def server(host, port, share, warmup):
    """Launch Gradio web interface."""
    from interface.app import main as app_main
    import sys

    args = [
        "--host", host,
        "--port", str(port),
    ]
    if share:
        args.append("--share")
    if warmup:
        args.append("--warmup")

    sys.argv = ["app.py"] + args
    app_main()


# ============================================================
# models
# ============================================================

@cli.group()
def models():
    """Model management commands."""
    pass


@models.command("list")
def models_list():
    """Show status of all models."""
    from src.model_utils import print_model_status
    print_model_status()


@models.command("download")
@click.option("--pair", default=None, help="Specific pair, e.g. en-vi")
@click.option("--all", "download_all", is_flag=True, help="Download all models")
@click.option("--convert-ct2", is_flag=True, help="Also convert to CTranslate2 INT8")
@click.option("--quantization", default="int8", show_default=True,
              type=click.Choice(["int8", "int8_float16", "float16", "float32"]))
def models_download(pair, download_all, convert_ct2, quantization):
    """Download pretrained models from HuggingFace."""
    from src.model_utils import download_model, download_all_models, convert_to_ct2, convert_all_to_ct2

    if download_all:
        console.print("[cyan]Downloading all models...[/cyan]")
        download_all_models()
        if convert_ct2:
            console.print("[cyan]Converting to CTranslate2...[/cyan]")
            convert_all_to_ct2(quantization=quantization)
    elif pair:
        console.print(f"[cyan]Downloading {pair}...[/cyan]")
        path = download_model(pair)
        console.print(f"[green]✓[/green] {path}")
        if convert_ct2:
            out = convert_to_ct2(pair, quantization=quantization)
            console.print(f"[green]✓[/green] CT2: {out}")
    else:
        console.print("[yellow]Specify --pair <pair> or --all[/yellow]")


@models.command("convert")
@click.option("--pair", required=True, help="Language pair, e.g. en-vi")
@click.option("--quantization", default="int8", show_default=True,
              type=click.Choice(["int8", "int8_float16", "float16", "float32"]))
def models_convert(pair, quantization):
    """Convert a model to CTranslate2 format."""
    from src.model_utils import convert_to_ct2
    out = convert_to_ct2(pair, quantization=quantization)
    console.print(f"[green]✓[/green] Saved to {out}")


# ============================================================
# finetune
# ============================================================

@cli.command()
@click.option("--pair", default="en-vi", show_default=True)
@click.option("--base-model", default=None)
@click.option("--train-tsv", default=None, type=click.Path(exists=True))
@click.option("--train-src", default=None, type=click.Path(exists=True))
@click.option("--train-tgt", default=None, type=click.Path(exists=True))
@click.option("--use-opus", is_flag=True)
@click.option("--epochs", default=3, show_default=True)
@click.option("--batch-size", default=8, show_default=True)
@click.option("--lr", default=5e-5, show_default=True)
@click.option("--max-samples", default=100_000, show_default=True)
@click.option("--fp16", is_flag=True)
@click.option("--output-dir", default=None)
def finetune(pair, base_model, train_tsv, train_src, train_tgt,
             use_opus, epochs, batch_size, lr, max_samples, fp16, output_dir):
    """Fine-tune a MarianMT model on custom data."""
    from src.finetune import FinetuneConfig, finetune as run_finetune

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = FinetuneConfig(
        pair_id=pair,
        base_model=base_model,
        train_tsv=train_tsv,
        train_src_file=train_src,
        train_tgt_file=train_tgt,
        use_opus=use_opus,
        max_train_samples=max_samples,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        fp16=fp16,
        output_dir=output_dir,
    )
    out = run_finetune(cfg)
    console.print(f"\n[green]Fine-tuned model saved to: {out}[/green]")


# ============================================================
# train-custom  (multilingual from scratch)
# ============================================================

@cli.command("train-custom")
@click.option("--config", "config_file", default="configs/multilingual.yaml",
              show_default=True, help="YAML config file")
@click.option("--output-dir", default=None)
@click.option("--epochs", default=None, type=int)
@click.option("--batch-size", default=None, type=int)
@click.option("--accum-steps", default=None, type=int)
@click.option("--lr", default=None, type=float, help="Learning rate (overrides config)")
@click.option("--warmup", default=None, type=int)
@click.option("--max-eval-samples", default=2000, show_default=True,
              help="Limit eval samples for speed (0=use all)")
@click.option("--eval-steps", default=None, type=int)
@click.option("--vocab-size", default=None, type=int)
@click.option("--spm-model", default=None)
@click.option("--en-vi", "en_vi", default=None, type=int, metavar="N")
@click.option("--zh-vi", "zh_vi", default=None, type=int, metavar="N")
@click.option("--ja-vi", "ja_vi", default=None, type=int, metavar="N")
@click.option("--ko-vi", "ko_vi", default=None, type=int, metavar="N")
@click.option("--no-reverse", is_flag=True)
@click.option("--temperature", default=None, type=float)
@click.option("--fp16", is_flag=True, default=None)
@click.option("--resume", default=None)
@click.option("--convert-ct2", is_flag=True)
def train_custom(config_file, output_dir, epochs, batch_size, accum_steps,
                 lr, warmup, max_eval_samples, eval_steps, vocab_size,
                 spm_model, en_vi, zh_vi, ja_vi, ko_vi,
                 no_reverse, temperature, fp16, resume, convert_ct2):
    """Train a single multilingual MarianMT model (EN/ZH/JA/KO ↔ VI).
    Loads configs/multilingual.yaml by default; CLI flags override YAML values.
    """
    from src.multilingual_train import MultilingualTrainConfig, train as run_train
    from src.translate_engine import convert_custom_to_ct2
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Load YAML config as base
    if config_file and os.path.isfile(config_file):
        cfg = MultilingualTrainConfig.from_yaml(config_file)
        console.print(f"[dim]Loaded config: {config_file}[/dim]")
    else:
        cfg = MultilingualTrainConfig()
        console.print("[yellow]No config file found, using defaults[/yellow]")

    # Apply CLI overrides (only when explicitly provided by user)
    if output_dir:       cfg.output_dir = output_dir
    if epochs:           cfg.num_train_epochs = epochs
    if batch_size:       cfg.per_device_train_batch_size = batch_size
    if accum_steps:      cfg.gradient_accumulation_steps = accum_steps
    if lr is not None:   cfg.learning_rate = lr
    if warmup:           cfg.warmup_steps = warmup
    if eval_steps:       cfg.eval_steps = eval_steps
    if vocab_size:       cfg.vocab_size = vocab_size
    if spm_model:        cfg.spm_model = spm_model
    if no_reverse:       cfg.add_reverse = False
    if temperature:      cfg.temperature = temperature
    if fp16:             cfg.fp16 = True
    if resume:           cfg.resume_from_checkpoint = resume
    if en_vi or zh_vi or ja_vi or ko_vi:
        cfg.pair_configs = {
            "en-vi": en_vi or cfg.pair_configs.get("en-vi", 500_000),
            "zh-vi": zh_vi or cfg.pair_configs.get("zh-vi", 200_000),
            "ja-vi": ja_vi or cfg.pair_configs.get("ja-vi", 200_000),
            "ko-vi": ko_vi or cfg.pair_configs.get("ko-vi", 200_000),
        }

    cfg.max_eval_samples = max_eval_samples

    console.print(f"[cyan]LR={cfg.learning_rate}  warmup={cfg.warmup_steps}  "
                  f"batch={cfg.per_device_train_batch_size}  "
                  f"max_eval_samples={max_eval_samples}[/cyan]")

    out = run_train(cfg)
    console.print(f"\n[green]Model saved to: {out}[/green]")

    if convert_ct2:
        console.print("[cyan]Converting to CTranslate2 INT8 ...[/cyan]")
        convert_custom_to_ct2()
        console.print("[green]Done! Fast inference enabled.[/green]")


# ============================================================
# detect  (language detection)
# ============================================================

@cli.command()
@click.argument("text")
def detect(text):
    """Detect the language of a text string."""
    from src.translate_engine import detect_language
    lang = detect_language(text)
    from src.config import LANG_NAMES, LANG_FLAGS
    console.print(f"[bold]{LANG_FLAGS.get(lang,'')} {LANG_NAMES.get(lang, lang)}[/bold] ({lang})")


# ============================================================
# convert-ct2  (convert custom model)
# ============================================================

@cli.command("convert-custom")
@click.option("--quantization", default="int8", show_default=True,
              type=click.Choice(["int8", "int8_float16", "float16", "float32"]))
def convert_custom(quantization):
    """Convert the custom multilingual model to CTranslate2 format."""
    from src.translate_engine import convert_custom_to_ct2
    convert_custom_to_ct2(quantization=quantization)
    console.print("[green]✓ Custom model converted to CTranslate2[/green]")


# ============================================================
# evaluate
# ============================================================

@cli.command()
@click.option("--src", "-s", required=True,
              type=click.Choice(["en", "vi", "zh", "ja", "ko"]))
@click.option("--tgt", "-t", required=True,
              type=click.Choice(["en", "vi", "zh", "ja", "ko"]))
@click.option("--src-file", default=None, type=click.Path(exists=True))
@click.option("--ref-file", default=None, type=click.Path(exists=True))
@click.option("--flores", is_flag=True, help="Use FLORES-200 benchmark")
@click.option("--beam", default=4)
@click.option("--device", default="cpu")
def evaluate(src, tgt, src_file, ref_file, flores, beam, device):
    """Evaluate translation quality (BLEU, chrF, TER)."""
    from src.evaluate import evaluate_model, evaluate_flores

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    engine = get_engine(device)

    if flores:
        results = evaluate_flores(engine, src, tgt)
    elif src_file and ref_file:
        with open(src_file, encoding="utf-8") as f:
            src_texts = f.read().splitlines()
        with open(ref_file, encoding="utf-8") as f:
            ref_texts = f.read().splitlines()
        results, _ = evaluate_model(engine, src_texts, ref_texts, src, tgt)
    else:
        console.print("[red]Specify --flores or --src-file + --ref-file[/red]")
        sys.exit(1)

    table = Table(title=f"Evaluation: {src}→{tgt}", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    for k, v in results.items():
        table.add_row(k.upper(), str(v))
    console.print(table)


# ============================================================

if __name__ == "__main__":
    cli()
