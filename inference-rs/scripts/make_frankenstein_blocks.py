#!/usr/bin/env python3
"""
Build a block corpus from Firefox's Full-Page Translations benchmark page
(`translations-bencher-en.html`, a public-domain Frankenstein excerpt) so our
native block benchmark translates the *same* text, chunked the *same* way, as
the shipping Wasm path.

Firefox chunks the page into block-level DOM elements and issues one translate()
call per element (a paragraph), skipping `translate="no"` subtrees. We mirror
that: one block per block-level element, sentences split onto one line each (so
block-bench with `ssplit-mode: sentence` and our engine's batched block path see
identical sentence sets). Output is blank-line-delimited blocks, matching
make_blocks.py / the --blocks harness.

Sentence splitting is a light heuristic (Firefox uses ssplit-cpp internally);
boundaries can differ slightly, but the source word/token totals — the numerator
for words/s and tokens/s — do not, so the cross-comparison is faithful.

    inference-rs/scripts/make_frankenstein_blocks.py
    inference-rs/scripts/make_frankenstein_blocks.py --html /path/to/page.html
"""

import argparse
import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path

CRATE_DIR = Path(__file__).resolve().parent.parent
# The benchmark page ships in the Firefox tree.
FIREFOX_HTML = Path(
    "/Users/greg/dev/firefox/toolkit/components/translations/tests/browser/"
    "translations-bencher-en.html"
)
OUT = CRATE_DIR / "corpora/frankenstein-en.blocks.txt"

BLOCK_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"}


class BlockExtractor(HTMLParser):
    """Collect the text of each block-level element, skipping any subtree whose
    element (or an ancestor) carries translate="no"."""

    def __init__(self):
        super().__init__()
        self.blocks: list[str] = []
        self._buf: list[str] = []
        self._capturing = 0  # depth of the block element we're inside (0 = none)
        self._no_translate_depth = 0
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        self._depth += 1
        d = dict(attrs)
        if d.get("translate") == "no":
            self._no_translate_depth = self._depth
        if tag in BLOCK_TAGS and not self._no_translate_depth and not self._capturing:
            self._capturing = self._depth
            self._buf = []

    def handle_endtag(self, tag):
        if self._capturing and self._depth == self._capturing:
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if text:
                self.blocks.append(text)
            self._capturing = 0
        if self._no_translate_depth and self._depth == self._no_translate_depth:
            self._no_translate_depth = 0
        self._depth -= 1

    def handle_data(self, data):
        if self._capturing and not self._no_translate_depth:
            self._buf.append(data)


# Abbreviations that take a period without ending a sentence, so we don't split
# after them (Firefox's ssplit-cpp has equivalent handling).
ABBR = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "st",
    "vs",
    "etc",
    "no",
    "vol",
    "p",
    "pp",
    "fig",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}


def split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation followed by whitespace and an
    uppercase/opening-quote/digit start, then merge back fragments that split
    after a known abbreviation or a single-letter initial. Heuristic, but stable
    for prose; source word/token totals are unaffected by boundary choices."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+(?=[«"“(A-ZÀ-ÖØ-Þ0-9])', text)
    merged: list[str] = []
    for piece in raw:
        if merged:
            m = re.search(r"(\w+)\.$", merged[-1])
            if m and (
                m.group(1).lower() in ABBR or (len(m.group(1)) == 1 and m.group(1).isupper())
            ):
                merged[-1] = merged[-1] + " " + piece
                continue
        merged.append(piece)
    return [p.strip() for p in merged if p.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--html", default=str(FIREFOX_HTML), help="benchmark HTML page")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    html = Path(args.html)
    if not html.exists():
        raise SystemExit(f"[error] benchmark HTML not found at {html}")

    parser = BlockExtractor()
    parser.feed(html.read_text(encoding="utf-8"))

    blocks = [split_sentences(b) for b in parser.blocks]
    blocks = [b for b in blocks if b]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join("\n".join(s for s in b) for b in blocks) + "\n")

    n_sent = sum(len(b) for b in blocks)
    n_words = sum(len(s.split()) for b in blocks for s in b)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(str(out) + ".sha256").write_text(f"{sha}  {out.name}\n")

    print(f"wrote {out}")
    print(f"  blocks:    {len(blocks)}")
    print(f"  sentences: {n_sent}")
    print(f"  words:     {n_words} (whitespace)")
    print(f"  sha256:    {sha}")


if __name__ == "__main__":
    main()
