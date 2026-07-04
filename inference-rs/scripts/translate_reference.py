#!/usr/bin/env python3
"""
Translate text with the reference C++ translator-cli.

This runs the existing C++ inference engine so the Rust implementation (inference-rs)
can be validated against a known-good reference. It pipes the input text into
translator-cli using the decode config written by download_model.py.

Prerequisites:
  1. Build the engine:    task inference-build
  2. Download the model:  task rs:download-model -- <src> <trg>

Run directly:
    inference-rs/scripts/translate_reference.py en es --text "Hello World"

Or through the task wrapper:
    task rs:translate-reference -- en es --text "Hello World"
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import translate_common as common

# The reference C++ engine, built by `task inference-build`.
DEFAULT_TRANSLATOR_CLI = "inference/build/src/app/translator-cli"
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
    common.add_common_args(parser)
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

    src, trg, langs, config = common.resolve_config(args.models_dir, args.source, args.target)

    text = common.read_input_text(args) + "\n"

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
