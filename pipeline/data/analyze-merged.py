"""
Extract a 1000-line sample from each source-target dataset pair.

Usage:

    python3 pipeline/data/analyze-merged.py \
        --src en \
        --trg ru \
        --datasets_glob 'path/to/datasets/*.zst' \
        --artifacts ./artifacts
"""

import argparse
from dataclasses import dataclass
import random
from pathlib import Path
from glob import glob

from pipeline.common.downloads import read_lines, write_lines, count_lines
from pipeline.common.logging import get_logger

logger = get_logger(__file__)

# Every dataset has a shuffled sample stored in a file.
DATASET_SAMPLE_SIZE = 1000
# Write out 100 samples as it's easy to skim through them.
MERGED_MEDIUM_SAMPLE_SIZE = 100
# Write out 10 samples so that an LLM can easily analyze a file.
MERGED_SMALL_SAMPLE_SIZE = 15


@dataclass
class Dataset:
    src_path: Path
    trg_path: Path
    name: str
    sample_path: Path
    line_count: int

    def extract_sample(self, src: str, trg: str) -> None:
        # Make this sorting stable based on the dataset name.
        random.seed(self.name)

        logger.info(f"Extracting sample from:\n - {self.src_path}\n - {self.trg_path}")

        # Get a random list of indexes to sample.
        indexes = list(range(self.line_count))
        random.shuffle(indexes)
        sample_indexes = set(indexes[:DATASET_SAMPLE_SIZE])

        # Store the lines in memory if they are within the sample.
        lines = []
        with (
            read_lines(self.src_path) as src_lines,
            read_lines(self.trg_path) as trg_lines,
        ):
            for i, (src_line, trg_line) in enumerate(zip(src_lines, trg_lines)):
                if i in sample_indexes:
                    lines.append((src_line, trg_line))

        # Shuffle the sample again in memory, otherwise it will be in a potentially
        # biased order of the dataset.
        random.shuffle(lines)

        logger.info(f"Writing {DATASET_SAMPLE_SIZE} to: {self.sample_path}")
        with write_lines(self.sample_path, encoding="utf-8-sig") as outfile:
            for src_line, trg_line in lines:
                outfile.write(f"{src}: {src_line}")
                outfile.write(f"{trg}: {trg_line}")
                outfile.write("\n")


def write_merged_sample_summary(
    datasets: list[Dataset], artifacts: Path, sample_count: int
) -> None:
    merged_output = artifacts / f"corpus.sample.{sample_count}.txt"
    logger.info(f"Merging sample files into: {merged_output}")

    with write_lines(merged_output, encoding="utf-8-sig") as out:
        out.write("--------\n")
        for dataset in datasets:
            logger.info(f"Writing {sample_count} merged samples for {dataset.name}")
            out.write(f"Dataset: {dataset.name}\n")
            out.write(f"Lines: {dataset.line_count}\n")
            out.write(f"Sample: {sample_count}\n")
            out.write("--------\n\n")

            # There are 3 lines per sample.
            line_count = sample_count * 3.0
            with read_lines(dataset.sample_path, encoding="utf-8-sig") as lines:
                for i, line in enumerate(lines):
                    if i >= line_count:
                        break
                    out.write(line)

            out.write("--------\n")


def get_datasets(src: str, trg: str, datasets_glob: str, artifacts: Path):
    all_paths = sorted(glob(datasets_glob))
    src_paths = [Path(p) for p in all_paths if p.endswith(f"{src}.zst")]

    datasets: list[Dataset] = []
    for src_path in src_paths:
        name = Path(src_path.stem).stem
        trg_path = src_path.parent / f"{name}.{trg}.zst"
        if not trg_path.exists():
            raise Exception(
                f"No matching target dataset was found:\nsrc: {src_path}\ntrg: {trg_path}"
            )
        line_count_src = count_lines(src_path)
        line_count_trg = count_lines(trg_path)

        assert line_count_src == line_count_trg, (
            f"Mismatched line count: "
            f"{src_path.name} ({line_count_src}) vs "
            f"{trg_path.name} ({line_count_trg})"
        )

        sample_path = artifacts / f"{name}.sample.txt"

        datasets.append(
            Dataset(
                src_path=src_path,
                trg_path=trg_path,
                name=name,
                sample_path=sample_path,
                line_count=line_count_src,
            )
        )
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--src",
        type=str,
        help="The source locale",
    )

    parser.add_argument(
        "--trg",
        type=str,
        help="The target locale",
    )

    parser.add_argument(
        "--datasets_glob",
        type=str,
        help="A glob-style path to the mono datasets, e.g. /path/to/*.zst",
    )

    parser.add_argument(
        "--artifacts",
        type=Path,
        help="The path to the artifacts directory.",
    )

    args = parser.parse_args()

    src: str = args.src
    trg: str = args.trg
    datasets_glob: str = args.datasets_glob
    artifacts: Path = args.artifacts

    artifacts.mkdir(exist_ok=True)

    datasets = get_datasets(src, trg, datasets_glob, artifacts)
    for dataset in datasets:
        dataset.extract_sample(src, trg)

    write_merged_sample_summary(datasets, artifacts, MERGED_MEDIUM_SAMPLE_SIZE)
    write_merged_sample_summary(datasets, artifacts, MERGED_SMALL_SAMPLE_SIZE)


if __name__ == "__main__":
    main()
