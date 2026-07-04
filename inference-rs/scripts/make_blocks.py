#!/usr/bin/env python3
"""
Group the committed NLLB corpus into paragraph-sized blocks for the block
benchmark ([07-batched-inference.md]).

The NLLB corpus is already one sentence per line, so this only *groups* — no
sentence splitting. Output is blank-line-delimited: one sentence per line, an
empty line between blocks. That keeps block boundaries unambiguous (an empty line
is never a sentence), distinct from the per-sentence newline.

Block sizes come from a tiny, documented LCG (a pure integer recurrence —
reproducible across languages and Python versions, unlike `random`'s internals),
skewed toward small (≈1–8 sentences, like real paragraphs) by taking the min of
two draws.

    inference-rs/scripts/make_blocks.py            # -> corpora/nllb-en-fr.blocks.txt (+ .sha256)
    task rs:make-blocks
"""

import argparse
import hashlib
from pathlib import Path

CRATE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = CRATE_DIR / "corpora/nllb-en-fr.txt"
DEFAULT_OUT = CRATE_DIR / "corpora/nllb-en-fr.blocks.txt"

# Numerical Recipes LCG (period 2^32): x <- (A*x + C) mod 2^32. Portable/auditable.
_LCG_A, _LCG_C, _LCG_M = 1664525, 1013904223, 1 << 32


def lcg(seed: int):
    x = seed & 0xFFFFFFFF
    while True:
        x = (_LCG_A * x + _LCG_C) % _LCG_M
        yield x


def block_size(gen, lo: int, hi: int) -> int:
    # min of two uniform draws over [lo, hi] -> skewed toward the small end.
    span = hi - lo + 1
    return min(lo + next(gen) % span, lo + next(gen) % span)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min", type=int, default=1, help="min sentences per block")
    ap.add_argument("--max", type=int, default=8, help="max sentences per block")
    args = ap.parse_args()

    lines = [l for l in Path(args.corpus).read_text().splitlines() if l.strip()]
    gen = lcg(args.seed)
    blocks, i = [], 0
    while i < len(lines):
        n = block_size(gen, args.min, args.max)
        blocks.append(lines[i : i + n])
        i += n

    text = "\n\n".join("\n".join(b) for b in blocks) + "\n"
    out = Path(args.out)
    out.write_text(text)
    sha = hashlib.sha256(text.encode()).hexdigest()
    Path(str(out) + ".sha256").write_text(sha + "\n")

    sizes = [len(b) for b in blocks]
    print(f"[blocks] {len(blocks)} blocks from {len(lines)} sentences -> {out}")
    print(
        f"[blocks] sentences/block min/mean/max = "
        f"{min(sizes)}/{sum(sizes) / len(sizes):.1f}/{max(sizes)} (seed {args.seed})"
    )
    print(f"[blocks] sha256 {sha}")


if __name__ == "__main__":
    main()
