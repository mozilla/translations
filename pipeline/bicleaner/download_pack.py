#!/usr/bin/env python3
"""
Downloads bicleaner-ai model for a lanuage pair.
Fallbacks to the multilingual model if the lanuage pair is not supported.

Example:
    python download_pack.py \
        --src=en \
        --trg=ru \
        artifacts/bicleaner-model-en-ru.zst
"""

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from typing import Optional

from pipeline.common.downloads import compress_file
from pipeline.common.logging import get_logger
from pipeline.langs.codes import LangCode

logger = get_logger(__file__)


# bicleaner-ai-download downloads the latest models from Hugging Face / Github
# If a new model is released and you want to invalidate Taskcluster caches,
# change this file since it is a part of the cache digest
# The last model was added to https://huggingface.co/bitextor on Mar 11, 2024
def _run_download(src: str, trg: str, dir: str) -> subprocess.CompletedProcess:
    # use large multilingual models
    model_type = "full-large" if trg == "xx" else "full"
    return subprocess.run(
        ["bicleaner-ai-download", src, trg, model_type, dir], capture_output=True, check=False
    )


def _compress_dir(dir_path: str) -> str:
    logger.info(f"Compressing {dir_path}")

    tarball_path = f"{dir_path}.tar"
    with tarfile.open(tarball_path, "w") as tar:
        tar.add(dir_path, arcname=os.path.basename(dir_path))

    compressed_path = str(compress_file(tarball_path))

    return compressed_path


def check_result(result: subprocess.CompletedProcess):
    """Checks the return code, and outputs the stdout and stderr if it fails."""
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()


def download(src: LangCode, trg: LangCode, output_path: str) -> None:
    # use pipeline language labels for the archive path
    tmp_dir = os.path.join(tempfile.gettempdir(), f"bicleaner-ai-{src}-{trg}")

    if os.path.exists(tmp_dir):
        # A previous download attempt failed, remove the temporary files.
        shutil.rmtree(tmp_dir)

    os.mkdir(tmp_dir)
    # convert to bicleaner supported language codes
    src = src.bicleaner()
    trg = trg.bicleaner()

    # Attempt to download a model.
    # 1: src-trg
    # 2: trg-src
    # 3: multilingual model
    logger.info(f"Attempt 1 of 3: Downloading a model for {src}-{trg}")
    result = _run_download(src, trg, tmp_dir)

    meta_path = os.path.join(tmp_dir, "metadata.yaml")
    if os.path.exists(meta_path):
        check_result(result)
        logger.info(f"The model for {src}-{trg} is downloaded")
    else:
        src, trg = trg, src
        logger.info(f"Attempt 2 of 3. Downloading a model for {src}-{trg}")
        result = _run_download(src, trg, tmp_dir)

        if os.path.exists(meta_path):
            check_result(result)
            print(f"The model for {src}-{trg} is downloaded")
        else:
            logger.info("Attempt 3 of 3. Downloading the multilingual model en-xx")
            src = "en"
            trg = "xx"
            result = _run_download(src, trg, tmp_dir)

            if not os.path.exists(meta_path):
                check_result(result)
                raise Exception("Could not download the multilingual model")

            print(f"The model for {src}-{trg} is downloaded")

    pack_path = tmp_dir
    logger.info("Compress the downloaded pack.")
    pack_path = _compress_dir(pack_path)

    # Move to the expected path
    logger.info(f"Moving {pack_path} to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shutil.move(pack_path, output_path)
    logger.info("Done")


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument("--src", type=str, help="Source language code")
    parser.add_argument("--trg", type=str, help="Target language code")
    parser.add_argument(
        "output_path",
        type=str,
        help="Full output file or directory path for example artifacts/en-pt.zst",
    )

    parsed_args = parser.parse_args(args)

    download(
        src=LangCode(parsed_args.src),
        trg=LangCode(parsed_args.trg),
        output_path=parsed_args.output_path,
    )


if __name__ == "__main__":
    main()
