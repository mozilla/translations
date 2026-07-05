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
    task rs:translate -- en es --text "Hello world!"
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import translate_common as common

# The crate root (inference-rs/) holds Cargo.toml; scripts/ is one level down.
CRATE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = CRATE_DIR / "artifacts"


def summarize_dhat(report: Path, gmax_bytes: int | None) -> int:
    """Print a short human summary of a dhat report and return an exit code.

    Peak heap comes from dhat's own `At t-gmax` line (authoritative — the JSON
    only stores per-site local maxima, which don't sum to the global peak). The
    top site is the allocation point holding the most bytes at its peak; we walk
    past the allocator internals to the first `inference_rs` frame. Sanity check
    (non-cheatable): that top site must be the Wemb embedding table
    (weights.rs) — if it ever isn't, the memory profile changed and we surface it.
    """
    d = json.loads(report.read_text())
    pps, ftbl = d["pps"], d["ftbl"]
    top = max(pps, key=lambda p: p.get("mb", 0))
    frames = [ftbl[i] for i in top["fs"]]
    site = next((f for f in frames if "fxtranslate" in f), frames[0] if frames else "?")
    wemb_dominated = any(("weights" in f) or ("Wemb" in f) for f in frames)

    peak = gmax_bytes if gmax_bytes is not None else top.get("mb", 0)
    print("\n[memory] dhat report:", report, file=sys.stderr)
    print(f"[memory] peak heap (t-gmax): {peak:,} bytes", file=sys.stderr)
    print(f"[memory] top site: {top['mb']:,} bytes @ {site.strip()}", file=sys.stderr)
    if wemb_dominated:
        print("[memory] OK: peak dominated by the Wemb table (as expected)", file=sys.stderr)
        return 0
    print(
        "[memory] WARNING: top allocation site is NOT the Wemb table — the memory "
        "profile changed; investigate.",
        file=sys.stderr,
    )
    return 1


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
    parser.add_argument(
        "--memory-report",
        action="store_true",
        help=(
            "Build with the dhat-heap feature and write a dhat JSON heap report to "
            "inference-rs/artifacts/, plus a human summary (peak bytes, top site). "
            "Off by default so normal builds stay small."
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

    # `cargo run` builds on demand; the diagnostic binary translates stdin line
    # by line. `fast` selects the native engine config; the dhat profiler is
    # compiled in only under its feature (jemalloc is ceded to it).
    cmd = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "fxtranslate-oracle",
        "--manifest-path",
        str(CRATE_DIR / "Cargo.toml"),
    ]
    cmd += ["--features", "fast,dhat-heap"] if args.memory_report else ["--features", "fast"]
    cmd += ["--", "translate", str(model_cfg["model"]), str(src_vocab), str(trg_vocab)]

    # Shortlisting is opt-in (it hurts short-text quality). The CLI auto-finds the
    # lex*.bin beside the model when the flag is set.
    if args.shortlist:
        cmd.append("--shortlist")

    print(f"[run] {' '.join(cmd)}", file=sys.stderr)

    if not args.memory_report:
        result = subprocess.run(cmd, input=text, text=True)
        sys.exit(result.returncode)

    # Memory-report path: point the profiler at artifacts/, capture stderr to read
    # dhat's peak line, then summarize the JSON.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    report = ARTIFACTS_DIR / f"dhat-{_langs}.json"
    env = {**os.environ, "DHAT_OUT": str(report)}
    result = subprocess.run(cmd, input=text, text=True, env=env, stderr=subprocess.PIPE)
    sys.stderr.write(result.stderr)  # echo the build/translate/dhat output through
    if result.returncode != 0:
        sys.exit(result.returncode)

    m = re.search(r"At t-gmax:\s*([\d,]+)\s*bytes", result.stderr)
    gmax = int(m.group(1).replace(",", "")) if m else None
    sys.exit(summarize_dhat(report, gmax))


if __name__ == "__main__":
    main()
