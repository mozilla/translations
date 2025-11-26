#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
import sys

from opuscleaner.filters.clean_common import CHARS


MIN_LENGTH = 2  # minimum number of words in a sentence
MAX_LENGTH = 200  # maximum number of words in a sentence

RATIO_ALPHA_WORDS = 0.4  # minimum fraction of "real" words in a sentence
RATIO_ALPHA_CHARS = 0.5  # minimum fraction of alpha characters in a sentence


def main():
    args = parse_user_args()

    for i, line in enumerate(sys.stdin):
        src = line.strip()
        if not src:
            continue

        skip = clean_mono(src, args.lang)
        if skip:
            if args.debug:
                sys.stderr.write("{}\t{}\n".format(skip, src))
            continue
        sys.stdout.write("{}\n".format(src))


def clean_mono(src, lang):
    # TODO: move mono cleaning to OpusCleaner
    #  when it support this https://github.com/hplt-project/OpusCleaner/issues/141

    # treat individual characters as tokens for CJK
    src_toks = src.split() if lang not in {"zh", "ja", "ko"} else src
    src_len = len(src_toks)

    if not src_len:
        return "EMPTY"

    if src_len < MIN_LENGTH:
        return "TOO_SHORT"

    if src_len > MAX_LENGTH:
        return "TOO_LONG"

    if lang in CHARS:
        num_alpha = sum([1 if re.match(CHARS[lang], t, re.IGNORECASE) else 0 for t in src_toks])
        if num_alpha / float(src_len) < RATIO_ALPHA_WORDS:
            return "RATIO_ALPHA"

        char_alpha = len(re.findall(CHARS[lang], src, re.IGNORECASE))
        if char_alpha / float(len(src.replace(" ", ""))) < RATIO_ALPHA_CHARS:
            return "RATIO_CHARS"

    return None


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lang", default="en")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
