"""
Download and (optionally) convert all registered MarianMT models.

Usage:
    python scripts/download_models.py                   # download EN↔VI only
    python scripts/download_models.py --all             # all pairs + pivot models
    python scripts/download_models.py --convert-ct2     # also convert to INT8
    python scripts/download_models.py --pair en-vi --convert-ct2
"""

import sys
import os
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from src.model_utils import (
    download_model, download_all_models,
    convert_to_ct2, convert_all_to_ct2,
    print_model_status,
)
from src.config import LANGUAGE_PAIRS, PIVOT_MODELS


def main():
    parser = argparse.ArgumentParser(description="Download MarianMT models")
    parser.add_argument("--pair", default=None,
                        help="Single pair to download, e.g. en-vi")
    parser.add_argument("--all", action="store_true",
                        help="Download all pairs and pivot models")
    parser.add_argument("--convert-ct2", action="store_true",
                        help="Convert to CTranslate2 INT8 after download")
    parser.add_argument("--quantization", default="int8",
                        choices=["int8", "int8_float16", "float16", "float32"])
    parser.add_argument("--status", action="store_true",
                        help="Show model status and exit")
    args = parser.parse_args()

    if args.status:
        print_model_status()
        return

    if args.pair:
        print(f"Downloading {args.pair} ...")
        path = download_model(args.pair)
        print(f"✓ {path}")
        if args.convert_ct2:
            out = convert_to_ct2(args.pair, quantization=args.quantization)
            print(f"✓ CT2: {out}")

    elif args.all:
        print("Downloading all models ...")
        download_all_models(include_pivot=True)
        if args.convert_ct2:
            print(f"Converting to CTranslate2 ({args.quantization}) ...")
            convert_all_to_ct2(quantization=args.quantization)
        print_model_status()

    else:
        # Default: download EN↔VI only (minimal setup)
        for pair in ["en-vi", "vi-en"]:
            print(f"Downloading {pair} ...")
            path = download_model(pair)
            print(f"✓ {path}")
            if args.convert_ct2:
                out = convert_to_ct2(pair, quantization=args.quantization)
                print(f"✓ CT2: {out}")

        print("\nTip: Run with --all to download all language pairs.")
        print("Tip: Run with --convert-ct2 for 3-4x faster inference.\n")
        print_model_status()


if __name__ == "__main__":
    main()
