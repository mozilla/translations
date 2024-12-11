import argparse
import gzip
import io
import json
import os
import unicodedata
import zipfile
from pathlib import Path

import requests

from pipeline.common.datasets import shuffle_with_max_lines
from pipeline.common.downloads import stream_download_to_file

"""
Build a monolingual dataset based off of NLLB.

task build-mono-nllb -- sl
"""

DATA_PATH = (Path(__file__).parent / "../data/nllb").resolve()


def stream_lines_from_remote_zip(url, filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip:
        with zip.open(filename, force_zip64=True) as file:
            for line in file:
                yield line.decode("utf-8").strip()


def compute_hashes_in_parallel_data(parallel_path: Path, lang: str):
    """
    In order to de-duplicate sentences we can compute a hash and store it in memory. This makes
    it so that we don't have to store the full sentence in memory
    """
    sentence_hashes: set[int] = set()
    sentences_visited = 0

    with zipfile.ZipFile(parallel_path.open(), "r") as zip_ref:  # type: ignore
        with zip_ref.open(f"NLLB.en-{lang}.{lang}") as mono_file:
            for line_bytes in mono_file:
                sentences_visited += 1
                if sentences_visited % 1_000_000 == 0:
                    print(f"Sentence number {sentences_visited:,}")
                sentence_hashes.add(hash_line(line_bytes.decode("utf-8")))

    return sentence_hashes, sentences_visited


def hash_line(line: str) -> int:
    """
    Return a hash of a line. The line has its whitespace stripped and text representation
    normalized to ensure a consistent representation.
    """
    cleaned_line = unicodedata.normalize("NFC", line.strip())
    return hash(cleaned_line)


def filter_and_write_monolingual_data(
    mono_path: Path, output_gzip_path: Path, sentence_hashes: set[int]
):
    """
    Filtering is done with a set[int]. Seeing if a line is in the set should be O(1)
    in terms of time complexity. A set[int] was chosen (storing the hash) rather than
    a set[str], as the latter would retain the string in memory.
    """
    with gzip.open(mono_path, "rt", encoding="utf-8") as mono_file, gzip.open(
        output_gzip_path, "wt", encoding="utf-8"
    ) as output:
        discard_count = 0
        kept_count = 0
        for line in mono_file:
            if hash_line(line) not in sentence_hashes:
                kept_count += 1
                output.write(line)
            else:
                discard_count += 1
            if kept_count % 1_000_000 == 0:
                print(f"{kept_count:,} kept, {discard_count:,} discarded")

    return kept_count, discard_count


def build_dataset_sample(output_gzip_path: Path, sample_path: Path, dataset_name: str):
    """
    Outputs a sample of 1000 randomly sampled sentences from the dataset
    """
    byte_size = output_gzip_path.stat().st_size
    with gzip.open(output_gzip_path, "rt", encoding="utf-8") as line_stream:
        with sample_path.open("w", encoding="utf-8") as output:
            for line in shuffle_with_max_lines(
                line_stream=line_stream,
                seed=dataset_name,
                max_lines=1000,
                total_byte_size=byte_size,
            ):
                output.write(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("language", metavar="LANG", type=str, help="The two/three letter langtag")
    parser.add_argument(
        "--cleanup", action="store_true", help="Delete the intermediate data files"
    )

    args = parser.parse_args()
    lang: str = args.language

    os.makedirs(DATA_PATH, exist_ok=True)

    mono_file = f"{lang}.txt.gz"
    mono_path = DATA_PATH / mono_file
    mono_url = f"https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/{mono_file}"

    parallel_file = f"en-{lang}.txt.zip"
    parallel_path = DATA_PATH / parallel_file
    parallel_url = f"https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/{parallel_file}"

    output_gzip_path = DATA_PATH / f"nllb-mono-{lang}.txt.gz"
    sample_path = DATA_PATH / f"nllb-mono-{lang}.sample.txt"
    output_info_path = DATA_PATH / f"nllb-mono-{lang}.info.json"

    if output_gzip_path.exists():
        print(f"{output_gzip_path} exists")
    else:
        if mono_path.exists():
            print(f"{mono_file} exists")
        else:
            stream_download_to_file(mono_url, mono_path)

        if parallel_path.exists():
            print(f"{parallel_file} exists")
        else:
            stream_download_to_file(parallel_url, parallel_path)
            # zip contents:
            # ├── README
            # ├── LICENSE
            # ├── NLLB.en-sl.en
            # ├── NLLB.en-sl.sl
            # └── NLLB.en-sl.scores

        print("Compute a hash of all the sentences in the parallel data.")
        print(f"{parallel_path}")

        sentence_hashes, sentences_visited = compute_hashes_in_parallel_data(parallel_path, lang)

        print(f"There are {len(sentence_hashes):,} unique sentences out of {sentences_visited:,}")
        print(f'{(sentences_visited - len(sentence_hashes)):,} "{lang}" sentences were duplicated')

        print("Identifying and writing out monolingual data.")
        kept_count, discard_count = filter_and_write_monolingual_data(
            mono_path, output_gzip_path, sentence_hashes
        )

        print(f"Dataset created {output_gzip_path}")
        print(f"{kept_count:,} kept, {discard_count:,} discarded")

        with output_info_path.open() as file:
            data = {"sentences_kept": kept_count, "sentences_discarded": discard_count}
            json.dump(data, file, indent=2)

    if sample_path.exists():
        print(f"{sample_path} exists")
    else:
        print(f"Building a sample of the data: {sample_path}")
        build_dataset_sample(output_gzip_path, sample_path, f"nllb-mono-{lang}")

    if args.cleanup:
        print(f"Cleaning up {mono_path}")
        mono_path.unlink()
        print(f"Cleaning up {parallel_path}")
        parallel_path.unlink()


if __name__ == "__main__":
    main()
