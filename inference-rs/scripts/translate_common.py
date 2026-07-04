#!/usr/bin/env python3
"""
Shared helpers for the translate scripts.

`translate_reference.py` (the C++ reference) and `translate.py` (the Rust engine)
take the same `<source> <target> [--text ...]` interface and resolve the same
downloaded model directory, so that plumbing lives here.
"""

import argparse
import sys
from pathlib import Path

DEFAULT_MODELS_DIR = "data/models"
DEFAULT_SOURCE = "en"
DEFAULT_TARGET = "es"


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the source/target/--text/--models-dir arguments both scripts share."""
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_SOURCE,
        help=f"Source language code, e.g. en (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "target",
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


def resolve_config(models_dir: str, source: str, target: str) -> tuple[str, str, str, Path]:
    """Resolve the decode config for a language pair, erroring if it is missing.

    Returns `(src, trg, langs, config_path)`.
    """
    src, trg = source.lower(), target.lower()
    langs = f"{src}{trg}"
    config = Path(models_dir) / langs / f"config.{langs}.yml"
    if not config.exists():
        raise SystemExit(
            f"[error] decode config not found at {config}\n"
            f"  Download the model first with: task rs:download-model -- {src} {trg}"
        )
    return src, trg, langs, config


def read_input_text(args: argparse.Namespace) -> str:
    """The text to translate: `--text` if given, otherwise all of stdin."""
    return args.text if args.text is not None else sys.stdin.read()


def _yaml_list(value: str) -> list[str]:
    """Parse a small inline YAML list like `[a, b]` into `['a', 'b']`."""
    return [item.strip() for item in value.strip().strip("[]").split(",") if item.strip()]


def parse_model_config(config: Path) -> dict:
    """Parse the model/vocab/shortlist paths out of a bergamot decode config.

    Only the handful of keys the Rust engine needs are read. Relative entries
    are resolved against the config's directory (the configs use
    `relative-paths: true`). Returns `{"model": Path, "vocabs": [Path, ...],
    "shortlist": Path | None}`.
    """
    base = config.parent
    fields: dict[str, list[str]] = {}
    for line in config.read_text().splitlines():
        line = line.strip()
        for key in ("models", "vocabs", "shortlist"):
            if line.startswith(key + ":"):
                fields[key] = _yaml_list(line.split(":", 1)[1])

    def resolve(name: str) -> Path:
        p = Path(name)
        return p if p.is_absolute() else base / p

    models = fields.get("models", [])
    vocabs = fields.get("vocabs", [])
    shortlist_entries = [e for e in fields.get("shortlist", []) if e.lower() != "false"]

    if not models:
        raise SystemExit(f"[error] no `models:` entry in {config}")
    if not vocabs:
        raise SystemExit(f"[error] no `vocabs:` entry in {config}")

    return {
        "model": resolve(models[0]),
        "vocabs": [resolve(v) for v in vocabs],
        "shortlist": resolve(shortlist_entries[0]) if shortlist_entries else None,
    }
