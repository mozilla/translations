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

from pipeline.langs.codes import to_iso6391, iso6393_and_script_to_iso6391


def main():
    args = parse_user_args()
    expected_lang = to_iso6391(args.lang)

    # nllb model confuses cmn with yue, use openlid
    if expected_lang == "zh":
        BIN = "openlid-v2.bin"
        URL = "https://huggingface.co/laurievb/OpenLID-v2/resolve/main/model.bin"
    else:
        BIN = "nllb.bin"
        URL = "https://dl.fbaipublicfiles.com/nllb/lid/lid218e.bin"

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
        lang = lid[0][0].replace("__label__", "")
        # cmn -> zh, eng -> en etc.
        lang = iso6393_and_script_to_iso6391(lang)
        if lang == expected_lang:
            sys.stdout.write(line)


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--field", default=0, type=int, help="text field, default: 0")
    parser.add_argument("-l", "--lang", required=True, type=str, help="Language to keep")
    return parser.parse_args()


if __name__ == "__main__":
    main()
