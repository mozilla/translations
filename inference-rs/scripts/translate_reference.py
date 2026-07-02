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
    inference-rs/scripts/translate_reference.py en es --text "Hello World"

Or through the task wrapper:
    task inference-rs:translate-reference -- en es --text "Hello World"
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# The reference C++ engine, built by `task inference-build`.
DEFAULT_TRANSLATOR_CLI = "inference/build/src/app/translator-cli"
DEFAULT_MODELS_DIR = "data/models"
DEFAULT_SOURCE = "en"
DEFAULT_TARGET = "es"
DEFAULT_CPU_THREADS = 4

# Where reference traces are written (gitignored). Passing --trace with no path
# defaults to <TRACE_DIR>/<src><trg>.trace here.
DEFAULT_TRACE_DIR = "inference-rs/artifacts"
# argparse sentinel: --trace given as a bare flag (no explicit path).
_TRACE_DEFAULT = object()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source",
        type=str,
        nargs="?",
        default=DEFAULT_SOURCE,
        help=f"Source language code, e.g. en (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "target",
        type=str,
        nargs="?",
        default=DEFAULT_TARGET,
        help=f"Target language code, e.g. es (default: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--text",
        help="Text to translate. If omitted, text is read from stdin.",
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
    parser.add_argument(
        "--trace",
        nargs="?",
        const=_TRACE_DEFAULT,
        default=None,
        metavar="PATH",
        help=(
            "Record a reference trace of every graph node (the parity oracle for "
            "inference-rs). Writes a binary trace and a human-readable <PATH>.txt "
            f"manifest. With no PATH, defaults to {DEFAULT_TRACE_DIR}/<src><trg>.trace. "
            "For a compact, complete trace use a short --text and --cpu-threads 1."
        ),
    )
    args = parser.parse_args()

    translator_cli = Path(args.translator_cli)
    if not translator_cli.exists():
        raise SystemExit(
            f"[error] translator-cli not found at {translator_cli}\n"
            "  Build it first with: task inference-build"
        )

    src, trg = args.source.lower(), args.target.lower()
    langs = f"{src}{trg}"

    config = Path(args.models_dir) / langs / f"config.{langs}.yml"
    if not config.exists():
        raise SystemExit(
            f"[error] decode config not found at {config}\n"
            f"  Download the model first with: task inference-rs:download-model -- "
            f"{src} {trg}"
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

    # The trace recorder in the C++ engine is enabled purely by MARIAN_TRACE, so
    # no CLI flag is needed on translator-cli itself.
    env = os.environ.copy()
    if args.trace is not None:
        if args.trace is _TRACE_DEFAULT:
            trace_path = Path(DEFAULT_TRACE_DIR) / f"{langs}.trace"
        else:
            trace_path = Path(args.trace)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        env["MARIAN_TRACE"] = str(trace_path)
        print(f"[trace] {trace_path}  (+ {trace_path}.txt manifest)", file=sys.stderr)

    print(f"[run] {' '.join(cmd)}", file=sys.stderr)

    result = subprocess.run(cmd, input=text, text=True, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
