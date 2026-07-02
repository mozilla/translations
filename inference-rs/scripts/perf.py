#!/usr/bin/env python3
"""
Performance harness for the Rust inference-rs engine.

Two modes:

  timing (default): translate a corpus `--runs` times (after `--warmup` discarded
    runs), parse the per-sentence `[timing]` spans the CLI emits, and report the
    median (the middle run) and IQR (interquartile range — the 25th-to-75th
    percentile spread, i.e. the band the middle half of runs fell in; tighter =
    more reproducible) of two metrics: TTFT (time to first token) and
    steady-state tokens/second.

  --samply: record one run under `samply` into a Firefox Profiler `.json.gz` in
    artifacts/, for flame-graph / call-tree inspection. The corpus is repeated
    (`--samply-loops`) so the sampler collects enough samples.

Fair-comparison notes (see issues/10-perf-harness.md): the release binary is
pre-built so the build is excluded from timing, and the engine is single-threaded.
Compare variants by passing `--features lean-embed` etc.

Run directly:
    inference-rs/scripts/perf.py en fr --runs 10
    inference-rs/scripts/perf.py en fr --features lean-embed
    inference-rs/scripts/perf.py en fr --samply
Or via the task wrapper:
    task inference-rs:perf -- en fr --runs 10
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import translate_common as common

CRATE_DIR = Path(__file__).resolve().parent.parent
BIN = CRATE_DIR / "target" / "release" / "inference-rs"
ARTIFACTS_DIR = CRATE_DIR / "artifacts"


def build(features: str) -> None:
    cmd = ["cargo", "build", "--release", "--manifest-path", str(CRATE_DIR / "Cargo.toml")]
    if features:
        cmd += ["--features", features]
    print(f"[build] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def engine_cmd(model, src_vocab, trg_vocab, shortlist: bool, timing: bool) -> list[str]:
    cmd = [str(BIN), "translate", str(model), str(src_vocab), str(trg_vocab)]
    if shortlist:
        cmd.append("--shortlist")
    if timing:
        cmd.append("--timing")
    return cmd


def run_timed(cmd: list[str], corpus: str) -> list[dict]:
    """One corpus pass; return the per-sentence timing spans parsed from stderr."""
    p = subprocess.run(cmd, input=corpus, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"[perf] engine failed (exit {p.returncode}):\n{p.stderr[-800:]}")
    prefix = "[timing] "
    return [
        json.loads(line[len(prefix) :])
        for line in p.stderr.splitlines()
        if line.startswith(prefix)
    ]


def iqr(xs: list[float]) -> tuple[float, float]:
    """Interquartile range: (25th percentile, 75th percentile) — the band the
    middle half of the values fall in. Falls back to (min, max)/point for tiny
    samples where quartiles aren't meaningful."""
    if len(xs) < 2:
        return (xs[0], xs[0])
    if len(xs) < 4:
        return (min(xs), max(xs))
    q = statistics.quantiles(xs, n=4)
    return (q[0], q[2])


def timing_mode(args, model, src_vocab, trg_vocab, corpus: str, n_sent: int) -> None:
    cmd = engine_cmd(model, src_vocab, trg_vocab, args.shortlist, timing=True)
    tokps, ttft = [], []
    for i in range(args.warmup + args.runs):
        spans = run_timed(cmd, corpus)
        if not spans:
            sys.exit("[perf] no timing spans parsed — is this a --timing build of the CLI?")
        if i < args.warmup:
            continue
        decode_s = sum(s["decode_ms"] for s in spans) / 1000.0
        tokps.append(sum(s["tokens"] for s in spans) / decode_s if decode_s else 0.0)
        ttft.append(statistics.median(s["ttft_ms"] for s in spans))

    feat = args.features or "(default)"
    corpus_name = Path(args.corpus).stem
    print(
        f"\ncorpus: {corpus_name} ({n_sent} sent) | 1 thread | "
        f"runs={args.runs} warmup={args.warmup} | shortlist={args.shortlist} | features={feat}"
    )
    print(f"{'metric':16}{'median':>10}{'IQR (25–75%)':>20}")
    for label, xs, unit in (("TTFT (ms)", ttft, ""), ("tok/s", tokps, "")):
        lo, hi = iqr(xs)
        print(f"{label:16}{statistics.median(xs):>10.1f}{f'{lo:.1f}–{hi:.1f}':>20}{unit}")
    # Spell the metrics out — this output gets peer-reviewed by low-context readers.
    print(
        "\nlegend:\n"
        "  median        = the middle run (robust to one slow outlier)\n"
        "  IQR (25–75%)  = interquartile range: the spread of the middle half of runs\n"
        "                  (25th–75th percentile); a tight band means reproducible\n"
        "  TTFT (ms)     = time to first token (encode + first decode step)\n"
        "  tok/s         = steady-state decode throughput (generated tokens / decode time)"
    )


def samply_mode(args, model, src_vocab, trg_vocab, corpus: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.source}{args.target}" + ("-lean" if "lean-embed" in args.features else "")
    out = ARTIFACTS_DIR / f"perf-{tag}.json.gz"
    # Repeat the corpus so a sub-second release run yields enough samples.
    big = corpus * args.samply_loops
    cmd = ["samply", "record", "--save-only", "--no-open", "-o", str(out), "--"] + engine_cmd(
        model, src_vocab, trg_vocab, args.shortlist, timing=False
    )
    print(f"[samply] {' '.join(cmd)}  (corpus x{args.samply_loops})", file=sys.stderr)
    p = subprocess.run(cmd, input=big, text=True, stdout=subprocess.DEVNULL)
    if p.returncode != 0:
        sys.exit(f"[samply] failed (exit {p.returncode})")
    print(
        f"[samply] wrote {out}\n"
        f"  view: samply load {out}   (or drag onto https://profiler.firefox.com)",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", nargs="?", default="en")
    parser.add_argument("target", nargs="?", default="fr")
    parser.add_argument("--models-dir", default=common.DEFAULT_MODELS_DIR)
    parser.add_argument("--corpus", default=str(CRATE_DIR / "corpora/dev-en.txt"))
    parser.add_argument("--limit", type=int, default=0, help="cap sentences (0 = all)")
    parser.add_argument("--runs", type=int, default=10, help="measured runs (timing mode)")
    parser.add_argument("--warmup", type=int, default=1, help="discarded warmup runs")
    parser.add_argument("--shortlist", action="store_true")
    parser.add_argument(
        "--features", default="", help="cargo features to build with, e.g. lean-embed"
    )
    parser.add_argument(
        "--samply", action="store_true", help="record a Firefox profile instead of timing"
    )
    parser.add_argument(
        "--samply-loops", type=int, default=50, help="repeat the corpus N times for the samply run"
    )
    args = parser.parse_args()

    _src, _trg, _langs, config = common.resolve_config(args.models_dir, args.source, args.target)
    model_cfg = common.parse_model_config(config)
    vocabs = model_cfg["vocabs"]
    src_vocab = vocabs[0]
    trg_vocab = vocabs[1] if len(vocabs) > 1 else vocabs[0]
    model = model_cfg["model"]

    lines = [l for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    if args.limit:
        lines = lines[: args.limit]
    corpus = "\n".join(lines) + "\n"

    build(args.features)

    if args.samply:
        samply_mode(args, model, src_vocab, trg_vocab, corpus)
    else:
        timing_mode(args, model, src_vocab, trg_vocab, corpus, len(lines))


if __name__ == "__main__":
    main()
