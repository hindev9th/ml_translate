"""
Download parallel corpus data for training / fine-tuning.

Sources:
  - OPUS-100 (via HuggingFace datasets) — high quality, 1M sentences per pair
  - Save as TSV for use with finetune.py

Usage:
    python scripts/download_data.py --pair en-vi --max 100000
    python scripts/download_data.py --pair zh-en --split train --max 200000
    python scripts/download_data.py --flores vi  # FLORES-200 for eval
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from src.data_utils import download_opus, download_flores, save_tsv, filter_pairs, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Download parallel corpus data")
    parser.add_argument("--pair", default=None, help="Language pair, e.g. en-vi")
    parser.add_argument("--split", default="train",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max", type=int, default=200_000,
                        help="Max number of sentence pairs")
    parser.add_argument("--flores", default=None, metavar="LANG",
                        help="Download FLORES-200 for a language code (en/vi/zh/ja/ko)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    out_dir = args.output_dir or DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    if args.flores:
        print(f"Downloading FLORES-200 for {args.flores} ...")
        sentences = download_flores(args.flores)
        out_path = os.path.join(out_dir, f"flores_{args.flores}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(sentences))
        print(f"✓ {len(sentences)} sentences → {out_path}")
        return

    if not args.pair:
        print("Specify --pair (e.g. en-vi) or --flores <lang>")
        sys.exit(1)

    print(f"Downloading OPUS-100 {args.pair} ({args.split}, max={args.max:,}) ...")
    sources, targets = download_opus(args.pair, split=args.split, max_samples=args.max)
    sources, targets = filter_pairs(sources, targets)

    out_path = os.path.join(out_dir, f"opus100_{args.pair}_{args.split}.tsv")
    save_tsv(sources, targets, out_path)
    print(f"✓ {len(sources):,} pairs saved → {out_path}")
    print(f"\nUse for fine-tuning:")
    print(f"  python src/finetune.py --pair {args.pair} --train-tsv {out_path}")


if __name__ == "__main__":
    main()
