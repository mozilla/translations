#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Usage:
#   ./langid-fasttext.py < sents.txt > code-tab-sents.txt
#
# Installation:
#   pip3 install pybind11 fasttext --user
#
# Parallelize:
#   cat sents.txt | parallel --pipe -k -j16 --block 20M ./langid-fasttext.py > code-tab-sents.txt

import argparse
import os
import sys

import fasttext

from pipeline.langs.codes import LangCode


def main():
    args = parse_user_args()
    lang = LangCode(args.lang)

    # nllb model confuses cmn with yue, use openlid
    if lang.is_chinese():
        BIN = "openlid-v2.bin"
        URL = "https://huggingface.co/laurievb/OpenLID-v2/resolve/main/model.bin"
    else:
        BIN = "nllb.bin"
        URL = "https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin"
    sys.stderr.write(f"Using model {BIN}\n")

    mpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), BIN)
    if not os.path.exists(mpath):
        sys.stderr.write("Downloading model {} ...\n".format(URL))
        import urllib.request

        urllib.request.urlretrieve(URL, mpath)

    model = fasttext.load_model(mpath)

    for line in sys.stdin:
        fields = line.strip().split("\t")
        lid = model.predict(fields[args.field])
        # lid: (('__label__eng_Latn',), array([0.79722595]))
        lang_id = lid[0][0].replace("__label__", "")
        # cmn_Hant -> zh_hant, cmn_Hans -> zh, eng_Latn -> en etc.
        pipeline_lang = LangCode.from_fasttext(lang_id)
        if args.debug:
            sys.stderr.write(f"{lang}\t{pipeline_lang}\t{line}")

        if pipeline_lang == lang:
            sys.stdout.write(line)


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--field", default=0, type=int, help="text field, default: 0")
    parser.add_argument("-l", "--lang", required=True, type=str, help="Language to keep")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        required=False,
        help="Print identified languages to stderr",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
