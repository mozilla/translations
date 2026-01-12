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

from pipeline.langs.codes import to_iso6391


def main():
    args = parse_user_args()

    if args.lang == "zh_hant":
        BIN = "openlid-v2.bin"
        URL = "https://huggingface.co/laurievb/OpenLID-v2/resolve/main/model.bin"
    else:
        BIN = "lid.176.bin"
        URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/{}".format(BIN)

    mpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), BIN)
    if not os.path.exists(mpath):
        sys.stderr.write("Downloading model {} ...\n".format(URL))
        import urllib.request

        urllib.request.urlretrieve(URL, mpath)

    model = fasttext.load_model(mpath)

    for line in sys.stdin:
        fields = line.strip().split("\t")
        lid = model.predict(fields[args.field])

        if BIN == "openlid-v2.bin":
            lang = lid[0][0].split("_")[4]
            # cmn -> zh, eng -> en etc.
            lang = to_iso6391(lang)
        else:
            lang = lid[0][0][-2:]

        sys.stdout.write("{}\t{}".format(lang, line))


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--field", default=0, type=int, help="text field, default: 0")
    parser.add_argument("-l", "--lang", type=str, help="Language to detect")
    return parser.parse_args()


if __name__ == "__main__":
    main()
