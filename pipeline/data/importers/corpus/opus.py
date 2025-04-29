#!/usr/bin/env python3
"""
Import data from OPUS.

Example usage:

pipeline/data/importers/corpus/opus.py                       \\
  en                                       `# src`           \\
  ru                                       `# trg`           \\
  artifacts/corpus                         `# output_prefix` \\
  opus_ELRC-3075-wikipedia_health/v1       `# dataset`
"""

import argparse
import shutil
from pathlib import Path
import zipfile

from requests import HTTPError

from pipeline.common.downloads import stream_download_to_file, compress_file
from pipeline.common.logging import get_logger

logger = get_logger(__file__)


def download_opus_corpus(src: str, trg: str, output_prefix: str, dataset: str):
    print("Downloading opus corpus")

    name = dataset.split("/")[0]
    name_and_version = "".join(c if c.isalnum() or c in "-_ " else "_" for c in dataset)

    tmp_dir = Path(output_prefix).parent / "opus" / name_and_version
    tmp_dir.mkdir(parents=True, exist_ok=True)

    archive_path = tmp_dir / f"{name}.txt.zip"

    try:
        logger.info(f"Downloading opus corpus for {src}-{trg} to {archive_path}")
        stream_download_to_file(
            f"https://object.pouta.csc.fi/OPUS-{dataset}/moses/{src}-{trg}.txt.zip", archive_path
        )
        pair = f"{src}-{trg}"
    except HTTPError:
        logger.info("HTTP error, trying opposite direction")
        logger.info(f"Downloading opus corpus for {trg}-{src} to {archive_path}")
        stream_download_to_file(
            f"https://object.pouta.csc.fi/OPUS-{dataset}/moses/{trg}-{src}.txt.zip", archive_path
        )
        pair = f"{trg}-{src}"

    logger.info("Extracting directory")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    logger.info("Compressing output files")
    for lang in (src, trg):
        file_path = tmp_dir / f"{name}.{pair}.{lang}"
        compressed_path = compress_file(file_path, keep_original=False, compression="zst")
        output_path = f"{output_prefix}.{lang}.zst"
        compressed_path.rename(output_path)

    shutil.rmtree(tmp_dir)
    print("Done: Downloading opus corpus")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument("src", type=str)
    parser.add_argument("trg", type=str)
    parser.add_argument("output_prefix", type=str)
    parser.add_argument("dataset", type=str)

    args = parser.parse_args()
    logger.info(f"src:           {args.src}")
    logger.info(f"trg:           {args.trg}")
    logger.info(f"output_prefix: {args.output_prefix}")
    logger.info(f"dataset:      {args.dataset}")

    download_opus_corpus(args.src, args.trg, args.output_prefix, args.dataset)


if __name__ == "__main__":
    main()
