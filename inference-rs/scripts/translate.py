#!/usr/bin/env python3
"""
Translate text with the Rust inference-rs engine.

Resolves the downloaded model for a language pair (the same layout
`translate_reference.py` uses), then calls the Rust CLI via `cargo run` so a
build is dispatched as needed. Text is piped over stdin, so the model is loaded
once and each input line is translated in turn.

Run directly:
    inference-rs/scripts/translate.py en es --text "Hello world!"

Or through the task wrapper:
    task inference-rs:translate -- en es --text "Hello world!"
"""

import argparse
import subprocess
import sys
from pathlib import Path

import translate_common as common

# The crate root (inference-rs/) holds Cargo.toml; scripts/ is one level down.
CRATE_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common.add_common_args(parser)
    parser.add_argument(
        "--shortlist",
        action="store_true",
        help=(
            "Restrict the output vocabulary to the model's lexical shortlist. Off by "
            "default: it lowers quality on short inputs."
        ),
    )
    args = parser.parse_args()

    src, trg, _langs, config = common.resolve_config(args.models_dir, args.source, args.target)
    model_cfg = common.parse_model_config(config)

    vocabs = model_cfg["vocabs"]
    src_vocab = vocabs[0]
    trg_vocab = vocabs[1] if len(vocabs) > 1 else vocabs[0]

    text = common.read_input_text(args)
    if not text.endswith("\n"):
        text += "\n"

    # `cargo run` builds on demand; the CLI translates stdin line by line.
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "--manifest-path",
        str(CRATE_DIR / "Cargo.toml"),
        "--",
        "translate",
        str(model_cfg["model"]),
        str(src_vocab),
        str(trg_vocab),
    ]

    # Shortlisting is opt-in (it hurts short-text quality). The CLI auto-finds the
    # lex*.bin beside the model when the flag is set.
    if args.shortlist:
        cmd.append("--shortlist")

    print(f"[run] {' '.join(cmd)}", file=sys.stderr)

    result = subprocess.run(cmd, input=text, text=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
