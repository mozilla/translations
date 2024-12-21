#!/usr/bin/env python3
"""
Calculates alignments for a parallel corpus.

Some efficiency measures were implemented as it needs to process 500M sentences long corpus for the student model:
1. Tokenization with Moses with remapping the alignments back to whitespace based tokenization to reduce vocabulary size and improve accuracy
2. Using fast C++ Moses tokenizer
2. Parallelization with multiprocessing (tokenization and remapping)
3. Buffering on writing the output files to improve throughput


Example:
    BIN=bin SRC=ru TRG=en python pipeline/alignments/align.py \
        --corpus_src=fetches/corpus.ru.zst
        --corpus_trg=fetches/corpus.en.zst
        --output_path=artifacts/corpus.aln.zst
        --priors_input_path=fetches/corpus.priors
        --priors_output_path=artifacts/corpus.priors
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
from contextlib import ExitStack
from enum import Enum
from glob import glob
from typing import Dict, Optional

import zstandard
from tqdm import tqdm

from pipeline.alignments.tokenizer import tokenize, TokenizerType
from pipeline.common.datasets import decompress
from pipeline.common.logging import get_logger

logger = get_logger("alignments")


class Tokenization(Enum):
    spaces = "spaces"
    moses = "moses"
    icu = "icu"


def run(
    corpus_src: str,
    corpus_trg: str,
    output_path: str,
    tokenization: Tokenization,
    chunk_lines: int,
    output_tokenized: bool,
    priors_input_path: Optional[str],
    priors_output_path: Optional[str],
):
    bin = os.environ["BIN"]
    src = os.environ["SRC"]
    trg = os.environ["TRG"]

    tmp_dir = os.path.join(os.path.dirname(output_path), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    corpus_src = maybe_decompress(corpus_src)
    corpus_trg = maybe_decompress(corpus_trg)

    if tokenization == Tokenization.spaces:
        tokenized_src, tokenized_trg = corpus_src, corpus_trg
        output_aln = output_path
    else:
        ext = f".tok-{tokenization.value}"
        tokenized_src = (
            corpus_src[: corpus_src.rfind(".")] + ext + corpus_src[corpus_src.rfind(".") :]
        )
        tokenized_trg = (
            corpus_trg[: corpus_trg.rfind(".")] + ext + corpus_trg[corpus_trg.rfind(".") :]
        )
        output_aln = os.path.join(tmp_dir, "aln")

        if tokenization == Tokenization.moses:
            tokenizer = TokenizerType.fast_moses
        elif tokenization == Tokenization.icu:
            tokenizer = TokenizerType.icu
        else:
            raise ValueError(f"Unrecognized tokenization type {tokenization}")
        # C++ tokenizer can process 100k sentences per second on a single core,
        # so the chunks to parallelize things should be large enough to increase throughput
        tokenize(corpus_src, tokenized_src, src, sentences_per_chunk=500000, tokenizer=tokenizer)
        tokenize(corpus_trg, tokenized_trg, trg, sentences_per_chunk=500000, tokenizer=tokenizer)

    fwd_path, rev_path = align(
        corpus_src=tokenized_src,
        corpus_trg=tokenized_trg,
        priors_input_path=priors_input_path,
        tmp_dir=tmp_dir,
        chunk_lines=chunk_lines,
    )
    symmetrize(bin=bin, fwd_path=fwd_path, rev_path=rev_path, output_path=output_aln)

    if priors_output_path:
        write_priors(
            corpus_src=tokenized_src,
            corpus_trg=tokenized_trg,
            fwd_path=fwd_path,
            rev_path=rev_path,
            priors_output_path=priors_output_path,
        )

    if tokenization != Tokenization.spaces:
        if output_tokenized:
            logger.info("Saving tokenized corpus")
            # Copy tokenized corpus to output directory
            for file in tokenized_src, tokenized_trg:
                output_corpus = shutil.move(file, os.path.dirname(output_path))
                subprocess.check_call(["zstdmt", "-f", "--rm", output_corpus])
        else:
            # Remap alignments to whitespace based tokenization
            remapped_aln = os.path.join(tmp_dir, "aln.remapped")
            remap(corpus_src, corpus_trg, tokenized_src, tokenized_trg, output_aln, remapped_aln)
            output_aln = remapped_aln

    if output_path.endswith(".zst"):
        logger.info("Compressing final alignments")
        subprocess.check_call(["zstdmt", "--rm", output_aln])
        output_aln += ".zst"
    shutil.move(output_aln, output_path)


def maybe_decompress(file_path: str):
    if file_path.endswith(".zst"):
        return str(decompress(file_path, remove=True, logger=logger))
    return file_path


def align(
    corpus_src: str,
    corpus_trg: str,
    tmp_dir: str,
    chunk_lines: int,
    priors_input_path: Optional[str],
):
    import eflomal

    logger.info("Splitting corpus into parts")
    # align in chunks to prevent OOM
    # produces chunks of files, like "corpus.en.aa", "corpus.en.ab", "corpus.en.ac" etc.
    subprocess.check_call(["split", "--lines", str(chunk_lines), corpus_src, corpus_src + "."])
    subprocess.check_call(["split", "--lines", str(chunk_lines), corpus_trg, corpus_trg + "."])

    fwd_path = os.path.join(tmp_dir, "aln.fwd")
    rev_path = os.path.join(tmp_dir, "aln.rev")

    for src_part in sorted(glob(f"{corpus_src}.*")):
        suffix = src_part.split(".")[-1]
        logger.info(f"Processing part {suffix}")

        with ExitStack() as stack:
            if priors_input_path:
                logger.info(f"Using provided priors: {priors_input_path}")
                priors_input = stack.enter_context(open(priors_input_path, "r", encoding="utf-8"))
            else:
                priors_input = None

            src_input = stack.enter_context(open(f"{corpus_src}.{suffix}", "r", encoding="utf-8"))
            trg_input = stack.enter_context(open(f"{corpus_trg}.{suffix}", "r", encoding="utf-8"))

            logger.info("Calculating alignments...")
            # We use eflomal aligner.
            # It is less memory intensive than fast_align.
            # fast_align failed with OOM in a large white-space tokenized corpus
            aligner = eflomal.Aligner()
            aligner.align(
                src_input,
                trg_input,
                links_filename_fwd=f"{fwd_path}.{suffix}",
                links_filename_rev=f"{rev_path}.{suffix}",
                priors_input=priors_input,
                quiet=False,
                use_gdb=False,
            )

    # Merge alignments parts into one file
    with open(fwd_path, "w") as fwd_out:
        fwd_parts = sorted(glob(f"{fwd_path}.*"))
        logger.info(f"Merging alignments: {fwd_parts}")
        subprocess.check_call(["cat"] + fwd_parts, stdout=fwd_out)
    with open(rev_path, "w") as rev_out:
        rev_parts = sorted(glob(f"{rev_path}.*"))
        logger.info(f"Merging alignments: {rev_parts}")
        subprocess.check_call(["cat"] + rev_parts, stdout=rev_out)

    return fwd_path, rev_path


def symmetrize(bin: str, fwd_path: str, rev_path: str, output_path: str):
    """
    Symmetrize the forward and reverse alignments of the corpus.

    Alignments are generated in two directions, source to target, and target to source.
    This function symmetrizes them so that both directions share the same alignment information.
    It uses `atools` binary from `fast_align`
    """
    logger.info("Symmetrizing alignments...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Wrap the file with a compressor stream if it needs to be compressed
    with ExitStack() as stack:
        with (
            zstandard.ZstdCompressor().stream_writer(stack.enter_context(open(output_path, "wb")))
            if output_path.endswith(".zst")
            else stack.enter_context(open(output_path, "w", encoding="utf-8"))
        ) as stream:
            with subprocess.Popen(
                [
                    os.path.join(bin, "atools"),
                    "-i",
                    fwd_path,
                    "-j",
                    rev_path,
                    "-c",
                    "grow-diag-final-and",
                ],
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding="utf-8",
            ) as proc:
                for line in proc.stdout:
                    stream.write(line.encode("utf-8") if output_path.endswith(".zst") else line)

                proc.wait()
                # Check for any errors in the subprocess execution
                if proc.returncode != 0:
                    logger.error(f"atools exit code: {proc.returncode}")
                    raise subprocess.CalledProcessError(proc.returncode, proc.args)


def write_priors(
    corpus_src: str,
    corpus_trg: str,
    fwd_path: str,
    rev_path: str,
    priors_output_path: str,
):
    import eflomal

    logger.info("Calculating priors...")
    with ExitStack() as stack:
        src_input = stack.enter_context(open(corpus_src, "r", encoding="utf-8"))
        trg_input = stack.enter_context(open(corpus_trg, "r", encoding="utf-8"))
        fwd_f = stack.enter_context(open(fwd_path, "r", encoding="utf-8"))
        rev_f = stack.enter_context(open(rev_path, "r", encoding="utf-8"))
        priors_tuple = eflomal.calculate_priors(src_input, trg_input, fwd_f, rev_f)
        logger.info(f"Writing priors to {priors_output_path}...")
        priors_output = stack.enter_context(open(priors_output_path, "w", encoding="utf-8"))
        eflomal.write_priors(priors_output, *priors_tuple)


def remap(
    src_path: str,
    trg_path: str,
    tok_src_path: str,
    tok_trg_path: str,
    aln_path: str,
    output_aln_path: str,
) -> None:
    """
    Remaps alignments that were calculated for tokenized corpus to whitespace-tokenized ones.
    :param src_path: path to whitespace-tokenized sentences in source language
    :param trg_path: path to whitespace-tokenized sentences in target language
    :param tok_src_path: path to tokenized sentences in source language
    :param tok_trg_path: path to tokenized sentences in target language
    :param aln_path: path to the alignments calculated for tokenized corpus
    :param output_aln_path: path to output alignments file remapped to whitespace-tokenized corpus
    """
    logger.info("Remapping alignments to whitespace tokenization")

    with ExitStack() as stack:
        pool = stack.enter_context(multiprocessing.Pool(processes=multiprocessing.cpu_count()))
        # Buffering helps to minimize IO operations which speeds thing up significantly
        output = stack.enter_context(open(output_aln_path, "w", buffering=500000))

        lines = zip(
            stack.enter_context(open(src_path)),
            stack.enter_context(open(trg_path)),
            stack.enter_context(open(tok_src_path)),
            stack.enter_context(open(tok_trg_path)),
            stack.enter_context(open(aln_path)),
        )

        # send lines to worker processes in chunks
        for aln in tqdm(pool.imap(remap_line, lines, chunksize=10000), mininterval=10):
            output.write(aln)


def remap_line(params):
    """
    Remaps alignments for a single line in a corpus
    """
    src, trg, tok_src, tok_trg, aln = params
    src_map = map_indices(tok_src, src)
    trg_map = map_indices(tok_trg, trg)

    remapped_aln = []
    for pair in aln.split():
        src_idx, trg_idx = map(int, pair.split("-"))
        new_pair = (src_map[src_idx], trg_map[trg_idx])
        if new_pair not in remapped_aln:
            remapped_aln.append(new_pair)

    return " ".join([f"{idx1}-{idx2}" for idx1, idx2 in remapped_aln]) + "\n"


def map_indices(tok_sentence: str, orig_sentence: str) -> Dict[int, int]:
    """
    Map token indices from tokenized sentence to original sentence.
    :param tok_sentence: tokenized sentence
    :param orig_sentence: original sentence
    :return: Dictionary of indices that maps tokenized words to original words
    """
    tok_words = tok_sentence.split()
    orig_words = orig_sentence.split()
    tok_to_orig_indices = {}
    orig_idx = 0
    tok_idx = 0

    # For 'Hello, world!'
    # Map ['Hello', ',', 'world', '!'] [0, 1, 2, 3]
    # To ['Hello,', 'world!'] [0, 1]
    # tok -> orig: {0: 0, 1: 0, 2: 1, 3: 1}

    while orig_idx < len(orig_words):
        orig_word = orig_words[orig_idx]
        word = ""
        while tok_idx < len(tok_words) and word != orig_word:
            word += tok_words[tok_idx]
            tok_to_orig_indices[tok_idx] = orig_idx
            tok_idx += 1

        orig_idx += 1

    return tok_to_orig_indices


def main() -> None:
    logger.info(f"Running with arguments: {sys.argv}")
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--type", metavar="TYPE", type=str, help="Dataset type: mono or corpus")
    parser.add_argument(
        "--corpus_src",
        metavar="CORPUS_SRC",
        type=str,
        help="Full path to the source sentences in a parallel dataset. Supports decompression using zstd. "
        "For example `fetches/corpus.ru` or `fetches/corpus.ru.zst`",
    )
    parser.add_argument(
        "--corpus_trg",
        metavar="CORPUS_TRG",
        type=str,
        help="Full path to the target sentences in a parallel dataset. Supports decompression using zstd. "
        "For example `fetches/corpus.en` or `fetches/corpus.en.zst`",
    )
    parser.add_argument(
        "--output_path",
        metavar="OUTPUT_PATH",
        type=str,
        help="A full path to the output alignments file. It will be compressed if the path ends with .zst. "
        "For example artifacts/corpus.aln or artifacts/corpus.aln.zst",
    )
    parser.add_argument(
        "--priors_input_path",
        metavar="PRIORS_INPUT_PATH",
        type=str,
        default=None,
        help="A full path to the model priors calculated in advance. This can speed up generation.",
    )
    parser.add_argument(
        "--priors_output_path",
        metavar="PRIORS_OUTPUT_PATH",
        type=str,
        default=None,
        help="Calculate and save the model priors to the specified file path. "
        "The file will be compressed if it ends with .zst",
    )
    parser.add_argument(
        "--tokenization",
        metavar="TOKENIZATION",
        type=Tokenization,
        choices=list(Tokenization),
        default=Tokenization.spaces,
        help="Use the specified tokenization method. Default is `spaces` which means no tokenization will be applied. "
        "It remaps the alignments back to whitespace tokenized ones if another tokenization method is used.",
    )
    parser.add_argument(
        "--output_tokenized",
        metavar="OUTPUT_TOKENIZED",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Output tokenized corpus and do not remap alignments to whitespace based tokenization",
    )
    parser.add_argument(
        "--chunk_lines",
        metavar="CHUNK_LINES",
        type=int,
        # use env to override from tests
        default=int(os.getenv("ALN_CHUNK_LINES", "50000000")),
        help="Split corpus to chunks of N lines to calculate alignments on them separately. "
        "This helps with reducing the memory footprint. 100M by default.",
    )
    args = parser.parse_args()
    logger.info("Starting generating alignments.")
    run(
        corpus_src=args.corpus_src,
        corpus_trg=args.corpus_trg,
        output_path=args.output_path,
        tokenization=args.tokenization,
        chunk_lines=args.chunk_lines,
        output_tokenized=args.output_tokenized,
        priors_input_path=args.priors_input_path,
        priors_output_path=args.priors_output_path,
    )
    logger.info("Finished generating alignments.")


if __name__ == "__main__":
    main()
