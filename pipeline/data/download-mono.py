#!/usr/bin/env python3
"""
Downloads a monolingual dataset, shuffles it, and truncates it to a maximum amount of sentences.

Kinds:
   taskcluster/kinds/dataset/kind.yml

Example usage:

    pipeline/data/download-mono.py                  \\
        --dataset news-crawl_news.2021              \\
        --language en                               \\
        --max_sentences 100000000                   \\
        --artifacts $TASK_WORKDIR/artifacts

Artifacts:

    artifacts
    └── news.2021.en.zst
"""

import argparse
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

from importers.mono.hplt import download_hplt

from pipeline.common.datasets import Dataset, shuffle_with_max_lines
from pipeline.common.downloads import (
    get_download_size,
    read_lines,
    write_lines,
)
from pipeline.common.logging import get_logger
from pipeline.data.cjk import ChineseConverter, ChineseType

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
IMPORTERS_PATH = os.path.abspath(os.path.join(CURRENT_FOLDER, "mono"))

logger = get_logger(__file__)


def main(args_list: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument("--dataset", type=str, help="The key for the dataset")
    parser.add_argument("--language", type=str, help="The BCP 47 language tag of the dataset")
    parser.add_argument(
        "--max_sentences", type=int, help="The maximum number of sentences to retain"
    )
    parser.add_argument(
        "--hplt_min_doc_score",
        type=float,
        help="The minimum document score to filter datasets that include this metric",
        default=5.0,
    )
    parser.add_argument(
        "--hplt_max_characters",
        type=int,
        help="The maximum number of characters to merge lines in a document before writing. "
        "0 - preserve original lines of HPLT dataset",
        default=0,
    )
    parser.add_argument(
        "--artifacts", type=Path, help="The location where the dataset will be saved"
    )
    args = parser.parse_args(args_list)

    dataset = Dataset(args.dataset)

    file_destination: Path = args.artifacts / f"{dataset.file_safe_name()}.{args.language}.zst"

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Max Sentences: {args.max_sentences}")
    logger.info(f"Minimum Document Score Threshold: {args.hplt_min_doc_score}")
    logger.info(f"Artifacts: {args.artifacts}")
    logger.info(f"File Destination: {file_destination}")

    if not os.path.exists(args.artifacts):
        os.makedirs(args.artifacts)

    if dataset.importer == "hplt":
        if dataset.name != "mono/v2.0":
            raise ValueError("Only HPLT v2.0 is supported")
        download_hplt(
            language=args.language,
            hplt_min_doc_score=args.hplt_min_doc_score,
            max_characters=args.hplt_max_characters,
            max_lines=args.max_sentences,
            file_destination=file_destination,
        )

        return

    url = None
    if dataset.importer == "url":
        url = dataset.name
    elif dataset.importer == "news-crawl":
        url = f"http://data.statmt.org/news-crawl/{args.language}/{dataset.name}.{args.language}.shuffled.deduped.gz"
        logger.info("Downloading WMT newscrawl monolingual data")
        logger.info(url)
    elif dataset.importer == "opus":
        url = f"https://object.pouta.csc.fi/OPUS-{dataset.name}/mono/{args.language}.txt.gz"
        logger.info("Downloading OPUS monolingual data")
        logger.info(url)
    else:
        raise Exception(f'Unsupported importer "{dataset.importer}"')

    logger.info(f"URL: {url}")

    with ExitStack() as stack:
        outfile = stack.enter_context(write_lines(file_destination))
        lines = stack.enter_context(read_lines(url))

        for line in shuffle_with_max_lines(
            line_stream=lines,
            seed=dataset.name,
            max_lines=args.max_sentences,
            total_byte_size=get_download_size(url),
        ):
            outfile.write(line)

    # TODO: convert everything to Chinese simplified for now
    # TODO: https://github.com/mozilla/firefox-translations-training/issues/896
    if args.language == "zh":
        logger.info("Converting the output file to Chinese Simplified")
        chinese_converter = ChineseConverter()
        converted_path = file_destination.with_suffix(".converted.zst")
        stats = chinese_converter.convert_file(
            file_destination, converted_path, ChineseType.simplified
        )
        shutil.move(converted_path, file_destination)
        print(
            f"Converted {stats.script_conversion.converted} lines from {stats.script_conversion.visited} to Chinese Simplified"
        )
        stats.save_json()


if __name__ == "__main__":
    main()
