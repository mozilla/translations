#!/usr/bin/env python3
"""
Corpus A/B parity: inference-rs vs the reference translator-cli.

Feeds a corpus (one pre-split sentence per line) through both engines and
reports the greedy exact-match rate plus the mismatches. Both engines run with
the shortlist OFF by default — the production baseline — so the comparison
matches what ships; pass --shortlist to enable it on both sides.

Each engine loads once (text is piped over stdin), so this is two process
launches, not two per sentence.

Usage:
    inference-rs/scripts/parity.py en fr
    inference-rs/scripts/parity.py en es --corpus path/to/corpus.txt --limit 50

Or via the task wrapper:
    task rs:parity -- en fr
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import translate_common as common

CRATE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CRATE_DIR.parent
DEFAULT_TRANSLATOR_CLI = REPO_ROOT / "inference/build/src/app/translator-cli"
DEFAULT_CORPUS = CRATE_DIR / "corpora/dev-en.txt"


def reference_outputs(config: Path, text: str, cpu_threads: int, shortlist: bool) -> list[str]:
    """Run translator-cli over the corpus. With shortlist off, run against a temp
    config copy that drops the `shortlist:` line (relative paths still resolve,
    since the copy sits in the config's directory)."""
    cfg = config
    tmp_path = None
    if not shortlist:
        kept = [
            line
            for line in config.read_text().splitlines()
            if not line.strip().startswith("shortlist:")
        ]
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", dir=config.parent, delete=False)
        tmp.write("\n".join(kept) + "\n")
        tmp.close()
        tmp_path = Path(tmp.name)
        cfg = tmp_path

    cmd = [
        str(DEFAULT_TRANSLATOR_CLI),
        "--model-config-paths",
        str(cfg),
        "--cpu-threads",
        str(cpu_threads),
    ]
    try:
        result = subprocess.run(cmd, input=text, text=True, capture_output=True)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return result.stdout.splitlines()


def rust_outputs(model_cfg: dict, text: str, shortlist: bool) -> list[str]:
    vocabs = model_cfg["vocabs"]
    src_vocab = vocabs[0]
    trg_vocab = vocabs[1] if len(vocabs) > 1 else vocabs[0]
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
    if shortlist:
        cmd.append("--shortlist")
    result = subprocess.run(cmd, input=text, text=True, capture_output=True)
    return result.stdout.splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", nargs="?", default="en")
    parser.add_argument("target", nargs="?", default="fr")
    parser.add_argument("--models-dir", default=common.DEFAULT_MODELS_DIR)
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="one sentence per line")
    parser.add_argument("--limit", type=int, default=0, help="cap sentences (0 = all)")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--shortlist", action="store_true", help="enable the shortlist on both sides"
    )
    args = parser.parse_args()

    _src, _trg, langs, config = common.resolve_config(args.models_dir, args.source, args.target)
    model_cfg = common.parse_model_config(config)

    lines = [l for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    if args.limit:
        lines = lines[: args.limit]
    text = "\n".join(lines) + "\n"

    ref = reference_outputs(config, text, args.cpu_threads, args.shortlist)
    rust = rust_outputs(model_cfg, text, args.shortlist)

    if len(ref) != len(lines) or len(rust) != len(lines):
        print(
            f"[warn] line-count mismatch: {len(lines)} in, {len(ref)} ref, {len(rust)} rust "
            "(a sentence may have been split); comparing by index up to the shortest.",
            file=sys.stderr,
        )

    n = min(len(lines), len(ref), len(rust))
    matches = 0
    mismatches = []
    for i in range(n):
        if ref[i] == rust[i]:
            matches += 1
        else:
            mismatches.append((lines[i], ref[i], rust[i]))

    shortlist_state = "on" if args.shortlist else "off"
    print(f"parity {langs} (shortlist {shortlist_state}): {matches}/{n} exact")
    for src, r, m in mismatches:
        print(f"  src : {src}")
        print(f"  ref : {r}")
        print(f"  rust: {m}")
        print()

    # Tracking metric, not a gate — always exit 0.


if __name__ == "__main__":
    main()
