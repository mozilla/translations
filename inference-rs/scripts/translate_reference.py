#!/usr/bin/env python3
"""
Translate text with the reference C++ translator-cli.

This runs the existing C++ inference engine so the Rust implementation (inference-rs)
can be validated against a known-good reference. It pipes the input text into
translator-cli using the decode config written by download_model.py.

Prerequisites:
  1. Build the engine:    task inference-build
  2. Download the model:  task inference-rs:download-model -- <src> <trg>

Run directly:
    inference-rs/scripts/translate_reference.py --langs enes "Hello World"

Or through the task wrapper:
    task inference-rs:translate-reference -- --langs enes "Hello World"
"""

import argparse
import subprocess
import sys
from pathlib import Path

# The reference C++ engine, built by `task inference-build`.
DEFAULT_TRANSLATOR_CLI = "inference/build/src/app/translator-cli"
DEFAULT_MODELS_DIR = "data/models"
DEFAULT_LANGS = "enes"
DEFAULT_CPU_THREADS = 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to translate. If omitted, text is read from stdin.",
    )
    parser.add_argument(
        "--langs",
        default=DEFAULT_LANGS,
        help=f"Language pair directory under --models-dir, e.g. enes (default: {DEFAULT_LANGS})",
    )
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Root directory the model was downloaded into (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--translator-cli",
        default=DEFAULT_TRANSLATOR_CLI,
        help=f"Path to the built translator-cli binary (default: {DEFAULT_TRANSLATOR_CLI})",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=DEFAULT_CPU_THREADS,
        help=f"Number of CPU threads for translator-cli (default: {DEFAULT_CPU_THREADS})",
    )
    args = parser.parse_args()

    translator_cli = Path(args.translator_cli)
    if not translator_cli.exists():
        raise SystemExit(
            f"[error] translator-cli not found at {translator_cli}\n"
            "  Build it first with: task inference-build"
        )

    config = Path(args.models_dir) / args.langs / f"config.{args.langs}.yml"
    if not config.exists():
        raise SystemExit(
            f"[error] decode config not found at {config}\n"
            f"  Download the model first with: task inference-rs:download-model -- "
            f"{args.langs[:2]} {args.langs[2:]}"
        )

    if args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()
    text += "\n"

    cmd = [
        str(translator_cli),
        "--model-config-paths",
        str(config),
        "--cpu-threads",
        str(args.cpu_threads),
    ]
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)

    result = subprocess.run(cmd, input=text, text=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
