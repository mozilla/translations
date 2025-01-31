#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import argparse
import collections
import math
import re
import sys


def main():
    args = parse_args()

    if args.metric == "bleu":
        score_function = compute_bleu
    elif args.metric == "sacrebleu":
        global sacrebleu
        import sacrebleu

        score_function = compute_sacrebleu
    elif args.metric == "chrf":
        global sacrebleu
        import sacrebleu

        score_function = compute_chrf
    else:
        sys.stderr.write("Unrecognized metric: {}\n".format(args.metric))
        pass

    if args.toolkit == "marian":
        marian_best_bleu(args, score_function)
    elif args.toolkit == "t2t":
        t2t_best_bleu(args, score_function)
        pass


def t2t_best_bleu(args, score_function):
    for i, ref_line in enumerate(args.references):
        refs = ref_line.strip().split("\n")
        if args.debpe:
            refs = [re.sub(r"@@ +", "", r) for r in refs]
            pass
        texts = next(args.nbest).strip().split("\t")
        if args.debpe:
            texts = [re.sub(r"@@ +", "", t) for t in texts]
            pass
        refs = [r.split() for r in refs]
        scores = [score_function(refs, t.split()) for t in texts]
        best_txt = texts[scores.index(max(scores))]

        args.output.write("{}\n".format(best_txt))
        if args.debug:
            sys.stderr.write("{}: {}\n".format(i, scores))
            pass
        if i % 100000 == 0 and i > 0:
            sys.stderr.write("[{}]\n".format(i))
            pass
        pass


def marian_best_bleu(args, score_function):
    prev_line = None
    for i, ref_line in enumerate(args.references):
        refs = ref_line.strip().split("\n")
        if args.debpe:
            refs = [re.sub(r"@@ +", "", r) for r in refs]

        texts = []
        while True:
            if prev_line:
                fields = prev_line.rstrip().split(" ||| ")

                # ctranslate2 can output empty text, for example:
                # 10181 ||| .GDFMAKERPROJECTファイルを開くには?
                # 10181 ||| .GDMAKERPROJECTファイルを開くには?
                # 10181 ||| .GDFMAKERPROJECTファイルを開くには?
                # 10181 ||| .GDFMakerPROJECTファイルを開くには?
                # 10181 ||| .GDFAKERPROJECTファイルを開くには?
                # 10181 ||| .GDMakerPROJECTファイルを開くには?
                # 10181 ||| .GDFMAKERPROJECTファイルを開くには。
                # 10181 |||
                if len(fields) == 2:
                    idx = int(fields[0])
                    if idx == i:
                        texts.append(fields[1])
                    else:
                        break

            prev_line = next(args.nbest, None)
            if not prev_line:
                break

        if args.debpe:
            texts = [re.sub(r"@@ +", "", t) for t in texts]
        refs = [r.split() for r in refs]
        scores = [score_function(refs, t.split()) for t in texts]
        best_txt = texts[scores.index(max(scores))]

        args.output.write("{}\n".format(best_txt))
        if args.debug:
            sys.stderr.write("{}: {}\n".format(i, scores))

        if i % 100000 == 0 and i > 0:
            sys.stderr.write("[{}]\n".format(i))


def compute_chrf(references, translation):
    hypo = " ".join(translation)
    refs = [" ".join(r) for r in references]
    return sacrebleu.sentence_chrf(hypo, refs).score


def compute_sacrebleu(references, translation):
    hypo = " ".join(translation)
    refs = [" ".join(r) for r in references]
    return sacrebleu.sentence_bleu(hypo, refs).score


def compute_bleu(references, translation, max_order=4):
    precisions = get_ngram_precisions(references, translation, max_order)
    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    bp = get_brevity_penalty(references, translation)
    return geo_mean * bp


def get_brevity_penalty(references, translation):
    reference_length = min(len(r) for r in references)
    translation_length = len(translation)
    ratio = float(translation_length) / reference_length
    if ratio > 1.0 or ratio == 0.0:
        bp = 1.0
    else:
        bp = math.exp(1 - 1.0 / ratio)
    return bp


def get_ngram_precisions(references, translation, max_order=4):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
        merged_ref_ngram_counts |= get_ngrams(reference, max_order)
    translation_ngram_counts = get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
        matches_by_order[len(ngram) - 1] += overlap[ngram]
    for order in range(1, max_order + 1):
        possible_matches = len(translation) - order + 1
        if possible_matches > 0:
            possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        # smoothing
        if matches_by_order[i] == 0 and possible_matches_by_order[i] == 0:
            precisions[i] = 0.0
        else:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
    return precisions


def get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i : i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def parse_args():
    from argparse import FileType

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--nbest", type=FileType("r"), default=sys.stdin)
    parser.add_argument("-r", "--references", type=FileType("r"), required=True)
    parser.add_argument("-o", "--output", type=FileType("w"), default=sys.stdout)
    parser.add_argument("-m", "--metric", default="bleu")
    parser.add_argument("--debpe", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-t", "--toolkit", default="marian", help="Toolkit: 'marian' or 't2t'")
    return parser.parse_args()


if __name__ == "__main__":
    main()
