#!/usr/bin/env python3
"""
Sample a diverse source-sentence corpus from NLLB / CCMatrix.

The parity and tokenizer oracles want real-world, diverse input. The full
CCMatrix bitext (the mined data behind NLLB) is tens of GB per pair, so this
streams the gzip, caps the download at ~500 MB, and reservoir-samples a fixed
subset (seeded) that gets committed under corpora/ with a sha256. The big file
is never fully downloaded or committed.

Source: https://www.statmt.org/cc-matrix/<a>-<b>.bitextf.tsv.gz
Rows are `score \\t <lang-a sentence> \\t <lang-b sentence>`, sorted by score
descending — so the sampled prefix is margin-biased toward high-confidence
pairs. That's the accepted "good enough" tradeoff: a single gzip stream can't be
seeked, so we sample the prefix rather than the whole file.

Usage:
    inference-rs/scripts/sample_corpus.py en fr           # writes corpora/nllb-en-fr.txt
    inference-rs/scripts/sample_corpus.py en es --n 1000 --max-mb 500 --seed 7

Or via the task wrapper:
    task rs:sample-corpus -- en fr
"""

import argparse
import gzip
import hashlib
import io
import random
import re
import sys
import urllib.request
from pathlib import Path

# A sentence boundary mid-string (terminal punctuation followed by space). Used
# to keep the corpus to single sentences, since the engine's contract is
# pre-split input and translator-cli would otherwise sentence-split and misalign
# the A/B comparison.
INTERNAL_BOUNDARY = re.compile(r'[.!?][\s")]')

CRATE_DIR = Path(__file__).resolve().parent.parent
CC_MATRIX_URL = "https://www.statmt.org/cc-matrix/{a}-{b}.bitextf.tsv.gz"


class CappedReader:
    """Wrap a stream so reads stop after `max_bytes` (of the compressed data)."""

    def __init__(self, stream, max_bytes: int):
        self.stream = stream
        self.remaining = max_bytes

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        want = self.remaining if size < 0 else min(size, self.remaining)
        chunk = self.stream.read(want)
        self.remaining -= len(chunk)
        return chunk


def acceptable(sentence: str, min_chars: int, max_chars: int) -> bool:
    n = len(sentence)
    if not (min_chars <= n <= max_chars) or "\t" in sentence:
        return False
    # Reject lines that look like more than one sentence (boundary before the end).
    return INTERNAL_BOUNDARY.search(sentence[:-1]) is None


def sample(
    url: str, max_bytes: int, n: int, seed: int, column: int, min_chars: int, max_chars: int
):
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen_unique = set()
    considered = 0

    req = urllib.request.Request(url, headers={"User-Agent": "inference-rs-sampler"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        capped = CappedReader(resp, max_bytes)
        gz = gzip.GzipFile(fileobj=capped)
        reader = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
        # Capping the compressed stream truncates the last gzip member, so a
        # trailing EOFError/OSError is expected — treat it as end-of-input.
        try:
            for raw in reader:
                cols = raw.rstrip("\n").split("\t")
                if len(cols) <= column:
                    continue
                sentence = cols[column].strip()
                if not acceptable(sentence, min_chars, max_chars) or sentence in seen_unique:
                    continue
                seen_unique.add(sentence)
                # Standard reservoir sampling: uniform over every sentence seen.
                considered += 1
                if len(reservoir) < n:
                    reservoir.append(sentence)
                else:
                    j = rng.randint(0, considered - 1)
                    if j < n:
                        reservoir[j] = sentence
        except (EOFError, OSError):
            pass
    return reservoir, considered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", help="source language code, e.g. en")
    parser.add_argument("target", help="target language code, e.g. fr")
    parser.add_argument("--n", type=int, default=1000, help="sentences to keep (default 1000)")
    parser.add_argument(
        "--max-mb", type=int, default=500, help="compressed download cap (default 500)"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=200)
    parser.add_argument(
        "--out", default=None, help="output path (default corpora/nllb-<src>-<trg>.txt)"
    )
    args = parser.parse_args()

    a, b = args.source.lower(), args.target.lower()
    url = CC_MATRIX_URL.format(a=a, b=b)
    # CCMatrix orders the pair alphabetically; the source column is the one
    # matching the requested source language.
    column = 1 if a < b else 2
    out = Path(args.out) if args.out else CRATE_DIR / "corpora" / f"nllb-{a}-{b}.txt"

    print(f"[sample] streaming {url}", file=sys.stderr)
    print(f"[sample] cap {args.max_mb} MB, keep {args.n}, seed {args.seed}", file=sys.stderr)
    reservoir, considered = sample(
        url, args.max_mb * 1024 * 1024, args.n, args.seed, column, args.min_chars, args.max_chars
    )
    if not reservoir:
        raise SystemExit("[error] no sentences sampled (bad pair, column, or network?)")

    # Deterministic order for a stable committed artifact.
    reservoir.sort()
    body = "\n".join(reservoir) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_text(f"{digest}  {out.name}\n")

    print(
        f"[sample] considered {considered} unique sentences (prefix, margin-biased); "
        f"kept {len(reservoir)} -> {out}",
        file=sys.stderr,
    )
    print(f"[sample] sha256 {digest}", file=sys.stderr)


if __name__ == "__main__":
    main()
