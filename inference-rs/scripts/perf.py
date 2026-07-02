#!/usr/bin/env python3
"""
Performance harness for the Rust inference-rs engine, cross-compared with the
marian-fork reference (translator-cli).

Each *run* processes the whole corpus through one already-loaded model, matching
production: the model is loaded once, sentences are translated one at a time in a
loop (no cross-sentence batching), then the model is unloaded when the process
exits. `--runs` such runs are timed (after `--warmup` discarded runs) and we
report the median and spread across runs.

  timing (default): translate a corpus with both engines and report, per engine:
    wall-clock seconds for the whole corpus and sentences/second (the fair
    cross-engine metric — both do the same one-off-per-sentence work on the same
    input). For inference-rs we additionally report TTFT and decode tok/s from
    the CLI's per-sentence `[timing]` spans. `--no-baseline` skips translator-cli.

  --samply: record one inference-rs run under `samply` into a Firefox Profiler
    `.json.gz` in artifacts/ (corpus repeated `--samply-loops` times for samples).

Fair-comparison notes (issues/10-perf-harness.md): the release binary is
pre-built so the build is excluded from timing; both engines run single-threaded;
both run shortlist-off (the production baseline) unless --shortlist.

CAVEAT on the marian baseline: bergamot's translator-cli is a *batch* tool — it
asserts mini-batch-words > max-length-break (128), so it cannot be reduced to one
sentence per batch. It runs at the production default (mini-batch-words 1024,
~30 sentences/batch), which batches across the whole corpus stream. That is an
UPPER BOUND on marian throughput and is NOT the one-off behavior of the Wasm path
(one document per call on a loaded model). A faithful one-off marian baseline
needs the wasm build; the translator-cli number is a conservative reference only.

Metrics are defined in a legend printed under the table — nothing cryptic, since
these numbers get peer-reviewed by low-context readers.

Run directly:
    inference-rs/scripts/perf.py en fr
    inference-rs/scripts/perf.py en fr --features lean-embed
    inference-rs/scripts/perf.py en fr --samply
Or via the task wrapper:
    task inference-rs:perf -- en fr
"""

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import translate_common as common

CRATE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CRATE_DIR.parent
BIN = CRATE_DIR / "target" / "release" / "inference-rs"
ARTIFACTS_DIR = CRATE_DIR / "artifacts"
DEFAULT_CORPUS = CRATE_DIR / "corpora/nllb-en-fr.txt"
DEFAULT_TRANSLATOR_CLI = REPO_ROOT / "inference/build/src/app/translator-cli"


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


def run_engine(cmd: list[str], corpus: str) -> tuple[float, list[dict]]:
    """One corpus pass: wall-clock seconds (load + translate all + exit) and the
    per-sentence timing spans parsed from stderr."""
    t0 = time.perf_counter()
    p = subprocess.run(cmd, input=corpus, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        sys.exit(f"[perf] inference-rs failed (exit {p.returncode}):\n{p.stderr[-800:]}")
    prefix = "[timing] "
    spans = [
        json.loads(line[len(prefix) :])
        for line in p.stderr.splitlines()
        if line.startswith(prefix)
    ]
    return wall, spans


def run_marian(config: Path, corpus: str, shortlist: bool, translator_cli: Path) -> float:
    """One corpus pass through translator-cli, one-off per sentence, single-threaded,
    model loaded once. Returns wall-clock seconds.

    NOTE: bergamot asserts mini-batch-words > max-length-break (128), so it cannot
    be forced to one sentence per batch — it is a batch tool. We therefore run it at
    the production/Wasm default (mini-batch-words: 1024) from the config verbatim,
    which batches across the corpus stream. This is an *upper bound* on marian
    throughput (more batching than the Wasm per-document path); a faithful one-off
    marian baseline needs the wasm build. Shortlist is dropped unless enabled."""
    kept = []
    for line in config.read_text().splitlines():
        s = line.strip()
        if s.startswith("shortlist:") and not shortlist:
            continue
        kept.append(line)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yml", dir=config.parent, delete=False)
    tmp.write("\n".join(kept) + "\n")
    tmp.close()
    tmp_path = Path(tmp.name)
    cmd = [str(translator_cli), "--model-config-paths", str(tmp_path), "--cpu-threads", "1"]
    try:
        t0 = time.perf_counter()
        p = subprocess.run(cmd, input=corpus, capture_output=True, text=True)
        wall = time.perf_counter() - t0
    finally:
        tmp_path.unlink(missing_ok=True)
    if p.returncode != 0:
        sys.exit(f"[perf] translator-cli failed (exit {p.returncode}):\n{p.stderr[-800:]}")
    return wall


def med_iqr(xs: list[float]) -> tuple[float, float, float]:
    """median, and interquartile range (25th, 75th percentile) — the band the
    middle half of the values fall in. Falls back to (min, max) for tiny samples."""
    m = statistics.median(xs)
    if len(xs) < 2:
        return (m, xs[0], xs[0])
    if len(xs) < 4:
        return (m, min(xs), max(xs))
    q = statistics.quantiles(xs, n=4)
    return (m, q[0], q[2])


def row(label: str, walls: list[float], n_sent: int) -> str:
    wm, wlo, whi = med_iqr(walls)
    sps = [n_sent / w for w in walls]
    sm, slo, shi = med_iqr(sps)
    return (
        f"{label:22}{wm:>8.2f}{f'{wlo:.2f}–{whi:.2f}':>16}"
        f"{sm:>10.1f}{f'{slo:.1f}–{shi:.1f}':>16}"
    )


def timing_mode(args, model, src_vocab, trg_vocab, config, corpus, n_sent) -> None:
    cmd = engine_cmd(model, src_vocab, trg_vocab, args.shortlist, timing=True)
    rs_walls, ttfts, dec_tokps = [], [], []
    for i in range(args.warmup + args.runs):
        wall, spans = run_engine(cmd, corpus)
        if not spans:
            sys.exit("[perf] no timing spans — is this a --timing build of the CLI?")
        if i < args.warmup:
            continue
        rs_walls.append(wall)
        ttfts.append(statistics.median(s["ttft_ms"] for s in spans))
        dec_s = sum(s["decode_ms"] for s in spans) / 1000.0
        dec_tokps.append(sum(s["tokens"] for s in spans) / dec_s if dec_s else 0.0)

    marian_walls = []
    if args.baseline:
        cli = Path(args.translator_cli)
        if not cli.exists():
            sys.exit(
                f"[perf] translator-cli not found at {cli} (build it, or pass --translator-cli)"
            )
        for i in range(args.warmup + args.runs):
            wall = run_marian(config, corpus, args.shortlist, cli)
            if i >= args.warmup:
                marian_walls.append(wall)

    feat = args.features or "default"
    print(
        f"\ncorpus: {Path(args.corpus).stem} ({n_sent} sent) | 1 thread | "
        f"runs={args.runs} warmup={args.warmup} | shortlist={'on' if args.shortlist else 'off'}\n"
        "  inference-rs: one-off (one sentence at a time, matching a production sentence iterator)\n"
        "  translator-cli: BATCHED (mini-batch-words 1024 — it can't go one-off; see note below)"
    )
    hdr = f"{'engine':22}{'wall (s)':>8}{'wall IQR':>16}{'sent/s':>10}{'sent/s IQR':>16}"
    print(hdr)
    print(row(f"inference-rs ({feat})", rs_walls, n_sent))
    if marian_walls:
        print(row("translator-cli", marian_walls, n_sent))
        rs_sps = n_sent / statistics.median(rs_walls)
        mar_sps = n_sent / statistics.median(marian_walls)
        print(f"{'ratio (rs / marian)':22}{'':>8}{'':>16}{rs_sps / mar_sps:>10.2f}x")

    ttft_m, ttft_lo, ttft_hi = med_iqr(ttfts)
    tok_m, tok_lo, tok_hi = med_iqr(dec_tokps)
    print(
        f"\ninference-rs internal (from --timing spans):\n"
        f"  TTFT (ms)      median {ttft_m:.1f}   IQR {ttft_lo:.1f}–{ttft_hi:.1f}\n"
        f"  decode tok/s   median {tok_m:.0f}   IQR {tok_lo:.0f}–{tok_hi:.0f}"
    )
    print(
        "\nlegend:\n"
        "  wall (s)     = seconds for one process to load the model, translate the whole\n"
        "                 corpus one sentence at a time, and exit (model loaded once)\n"
        "  sent/s       = sentences per second = corpus size / wall (fair cross-engine metric)\n"
        "  IQR          = interquartile range: 25th–75th percentile across runs (tight = stable)\n"
        "  TTFT (ms)    = time to first token per sentence (encode + first decode step)\n"
        "  decode tok/s = generated tokens / decode time (excludes model load; inference-rs only)\n"
        "  ratio        = inference-rs sent/s ÷ translator-cli sent/s (>1 = inference-rs faster)\n"
        "\nCAVEAT: translator-cli is a batch tool — bergamot asserts mini-batch-words >\n"
        "max-length-break (128), so it can't be forced to one sentence per batch. It runs\n"
        "here at the production default (1024 words, ~30 sentences/batch), which is an UPPER\n"
        "BOUND on marian throughput. A faithful one-off marian baseline (one document per\n"
        "call on a loaded model, like the Wasm path) needs the wasm build, not translator-cli."
    )


def samply_mode(args, model, src_vocab, trg_vocab, corpus: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.source}{args.target}" + ("-lean" if "lean-embed" in args.features else "")
    out = ARTIFACTS_DIR / f"perf-{tag}.json.gz"
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
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="one sentence per line")
    parser.add_argument("--limit", type=int, default=0, help="cap sentences (0 = all)")
    parser.add_argument("--runs", type=int, default=10, help="measured runs")
    parser.add_argument("--warmup", type=int, default=1, help="discarded warmup runs")
    parser.add_argument(
        "--shortlist", action="store_true", help="enable shortlist on both engines"
    )
    parser.add_argument(
        "--features", default="", help="cargo features to build with, e.g. lean-embed"
    )
    parser.add_argument(
        "--no-baseline",
        dest="baseline",
        action="store_false",
        help="skip the translator-cli cross-comparison",
    )
    parser.add_argument("--translator-cli", default=str(DEFAULT_TRANSLATOR_CLI))
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
        timing_mode(args, model, src_vocab, trg_vocab, config, corpus, len(lines))


if __name__ == "__main__":
    main()
