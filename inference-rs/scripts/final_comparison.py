#!/usr/bin/env python3
"""
Three-way apples-to-apples comparison on the SAME data and SAME model:

  1. inference-rs      — our Rust engine (default fast: lean-embed + gemmology)
  2. marian block-bench — the native marian-fork reference (bergamot)
  3. Firefox Wasm       — Full-Page Translations (numbers pasted from a perftest run)

All three translate Firefox's benchmark page (a Frankenstein excerpt) with the
en→ru **base** model, block by block (one batched translate per paragraph on a
loaded engine — the production shape). Metrics mirror Firefox's TranslationsBencher:

  words/s      = source words ÷ translation seconds        (higher better)
  tokens/s     = source spm tokens ÷ translation seconds   (higher better)
  translate s  = translation wall time, model load EXCLUDED (lower better)
  init ms      = model load / engine init                  (lower better)
  peak RSS MiB = peak resident memory of the process       (lower better)

`translation seconds` excludes model load on all three (Firefox measures
engine-ready → done; we sum per-block compute; init is measured separately).

Run:  inference-rs/scripts/final_comparison.py            (en→ru, base model)
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import translate_common as common

CRATE = Path(__file__).resolve().parent.parent
REPO = CRATE.parent
BIN = CRATE / "target/release/inference-rs"
BLOCK_BENCH = REPO / "inference/build/src/app/block-bench"
DEFAULT_BLOCKS = CRATE / "corpora/frankenstein-en.blocks.txt"

# Firefox "Full-Page Translations Base Model" (en→ru), medians of the 5-run
# perftest the user provided. wordCount/tokenCount are the page's source totals.
FIREFOX = {
    "label": "Firefox Wasm (Full-Page)",
    "words_per_second": 418.982,
    "tokens_per_second": 566.884,
    "translate_s": 22.853,  # total-translation-time (engine-ready → done)
    "init_ms": 135.169,  # engine-init-time
    "peak_rss_mib": 355.361,  # peak-inference-process-memory-usage
    "peak_parent_mib": 384.928,
    "word_count": 9575,
    "token_count": 12955,
}


def med(xs):
    return statistics.median(xs)


def time_wrap(cmd, stdin_text):
    """Run `cmd` under /usr/bin/time -l. Returns (wall_s, stderr, peak_rss_mib)."""
    t0 = time.perf_counter()
    p = subprocess.run(
        ["/usr/bin/time", "-l", *cmd], input=stdin_text, capture_output=True, text=True
    )
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        sys.exit(f"[final] command failed: {' '.join(cmd)}\n{p.stderr[-2000:]}")
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    rss = int(m.group(1)) / (1024 * 1024) if m else 0.0  # macOS reports bytes
    return wall, p.stderr, rss


def parse_blocks(stderr):
    pre = "[block] "
    return [json.loads(l[len(pre) :]) for l in stderr.splitlines() if l.startswith(pre)]


def marian_config(config: Path) -> Path:
    """Temp config: drop shortlist, force ssplit-mode: sentence (pre-split input)."""
    kept = [
        l
        for l in config.read_text().splitlines()
        if not l.strip().startswith(("shortlist:", "ssplit-mode:"))
    ]
    kept.append("ssplit-mode: sentence")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", dir=config.parent, delete=False)
    tmp.write("\n".join(kept) + "\n")
    tmp.close()
    return Path(tmp.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", nargs="?", default="en")
    ap.add_argument("target", nargs="?", default="ru")
    ap.add_argument("--models-dir", default=common.DEFAULT_MODELS_DIR)
    ap.add_argument("--blocks", default=str(DEFAULT_BLOCKS))
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    _s, _t, _l, config = common.resolve_config(args.models_dir, args.source, args.target)
    mc = common.parse_model_config(config)
    model, vocabs = mc["model"], mc["vocabs"]
    srcv = vocabs[0]
    trgv = vocabs[1] if len(vocabs) > 1 else vocabs[0]

    blocks = Path(args.blocks)
    text = blocks.read_text()
    n_blocks = sum(1 for c in text.split("\n\n") if c.strip())
    src_words = sum(len(l.split()) for l in text.splitlines() if l.strip())

    print(f"[final] building release (default fast config)…", file=sys.stderr)
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(CRATE / "Cargo.toml")],
        check=True,
    )
    if not BLOCK_BENCH.exists():
        sys.exit(f"[final] block-bench not found at {BLOCK_BENCH}")

    engine_cmd = [
        str(BIN),
        "translate",
        str(model),
        str(srcv),
        str(trgv),
        "--blocks",
        str(blocks),
        "--timing",
    ]
    tmpcfg = marian_config(config)
    marian_cmd = [str(BLOCK_BENCH), "--model-config-paths", str(tmpcfg)]

    # Accumulators across runs.
    rs = {"wps": [], "tps": [], "translate_s": [], "init_ms": [], "rss": []}
    mar = {"wps": [], "tps": [], "translate_s": [], "init_ms": [], "rss": []}
    src_tokens = None
    try:
        for i in range(args.warmup + args.runs):
            # inference-rs
            wall, err, rss = time_wrap(engine_cmd, None)
            spans = parse_blocks(err)
            compute_s = sum(s["encode_ms"] + s["decode_ms"] for s in spans) / 1000.0
            src_tokens = sum(s["src_tokens"] for s in spans)
            if i >= args.warmup:
                rs["wps"].append(src_words / compute_s)
                rs["tps"].append(src_tokens / compute_s)
                rs["translate_s"].append(compute_s)
                rs["init_ms"].append((wall - compute_s) * 1000.0)
                rs["rss"].append(rss)

            # marian block-bench
            wall, err, rss = time_wrap(marian_cmd, text)
            spans = parse_blocks(err)
            compute_s = sum(s["wall_ms"] for s in spans) / 1000.0
            if i >= args.warmup:
                mar["wps"].append(src_words / compute_s)
                mar["tps"].append(src_tokens / compute_s)
                mar["translate_s"].append(compute_s)
                mar["init_ms"].append((wall - compute_s) * 1000.0)
                mar["rss"].append(rss)
    finally:
        tmpcfg.unlink(missing_ok=True)

    print(
        f"\ncorpus: {blocks.stem} ({n_blocks} blocks, {src_words} source words, "
        f"{src_tokens} source tokens) | model: {model.parent.name} base | 1 thread | "
        f"shortlist off | runs={args.runs} warmup={args.warmup}\n"
    )
    hdr = f"{'engine':28}{'words/s':>9}{'tokens/s':>10}{'translate s':>13}{'init ms':>10}{'peak RSS MiB':>14}"
    print(hdr)

    def row(label, d):
        print(
            f"{label:28}{med(d['wps']):>9.0f}{med(d['tps']):>10.0f}"
            f"{med(d['translate_s']):>13.2f}{med(d['init_ms']):>10.0f}{med(d['rss']):>14.0f}"
        )

    row("inference-rs (rust, fast)", rs)
    row("marian block-bench (native)", mar)
    print(
        f"{FIREFOX['label']:28}{FIREFOX['words_per_second']:>9.0f}"
        f"{FIREFOX['tokens_per_second']:>10.0f}{FIREFOX['translate_s']:>13.2f}"
        f"{FIREFOX['init_ms']:>10.0f}{FIREFOX['peak_rss_mib']:>14.0f}"
    )

    rw, mw, fw = med(rs["wps"]), med(mar["wps"]), FIREFOX["words_per_second"]
    print(
        "\nspeedups (words/s):\n"
        f"  inference-rs vs Firefox Wasm : {rw / fw:.2f}x\n"
        f"  marian native vs Firefox Wasm: {mw / fw:.2f}x\n"
        f"  inference-rs vs marian native: {rw / mw:.2f}x"
    )
    print(
        "\nnotes:\n"
        "  - Same en→ru BASE model (dim-emb 512, ffn 2048, SSRU dec) and same\n"
        "    Frankenstein text for all three; block = paragraph = one translate call.\n"
        "  - translate s excludes model load on all three (Firefox: engine-ready→done;\n"
        "    native: summed per-block compute). init ms is model load / engine init.\n"
        "  - peak RSS: whole-process for inference-rs & block-bench; Firefox value is\n"
        "    its inference *subprocess* (peak-inference-process-memory-usage). Firefox\n"
        "    also runs a parent process (~%d MiB peak) that the native tools have no\n"
        "    equivalent of.\n"
        "  - words: whitespace (%d); Firefox counts %d via ICU. tokens: source spm\n"
        "    subwords (%d); Firefox counts %d. Same text, ~1%% counting differences.\n"
        "  - Firefox Wasm carries HTML parsing + process IPC per block that the native\n"
        "    harnesses do not; it is the shipping end-to-end path, not just the kernel."
        % (
            round(FIREFOX["peak_parent_mib"]),
            src_words,
            FIREFOX["word_count"],
            src_tokens,
            FIREFOX["token_count"],
        )
    )


if __name__ == "__main__":
    main()
