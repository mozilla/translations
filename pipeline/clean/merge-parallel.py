"""
Merges multiple parallel corpora into a single "source" language file, and a single
"target" language file, each.

For instance:

  dataset1.en.zst dataset1.ru.zst
  dataset2.en.zst dataset2.ru.zst
  dataset3.en.zst dataset3.ru.zst

  Gets merged into:

  corpus.en.zst
  corpus.ru.zst
"""

import argparse
from contextlib import ExitStack
from glob import glob
from pathlib import Path
from typing import Generator, Optional
from pipeline.common.datasets import (
    FilteringStep,
    Statistics,
    WeakStringSet,
    shuffle_with_max_lines,
)
from pipeline.common.downloads import get_human_readable_file_size, read_lines, write_lines
from pipeline.common.logging import get_logger

logger = get_logger(__file__)


class FilteringStatistics(Statistics):
    """
    Gather statistics about the filtering process.
    """

    def __init__(self, dataset_path: Path) -> None:
        super().__init__(dataset_path)
        self.parallel_corpus = FilteringStep(
            "The parallel corpora are merged and deduplicated",
        )
        self.final_truncated = FilteringStep("The final result can be truncated by max_lines")
        self.datasets = []

    def add_parallel_dataset(self, location: str):
        # e.g. /path/to/ada83_v1.en.zst
        path = Path(location)
        # e.g. ada83_v1
        dataset_stem = Path(path.stem).stem
        step = FilteringStep(dataset_stem)
        self.datasets.append(step)
        return step


def log_dataset(location: str):
    logger.info(f"Reading dataset {location}")


class DeduplicateCorpus:
    def __init__(
        self,
        datasets_src: list[Path],
        datasets_trg: list[Path],
        src_outpath: Path,
        trg_outpath: Path,
        stats: FilteringStatistics,
    ) -> None:
        self.datasets_src: list[Path] = datasets_src
        self.datasets_trg: list[Path] = datasets_trg
        self.src_outpath: Path = src_outpath
        self.trg_outpath: Path = trg_outpath
        self.stats: FilteringStatistics = stats
        self.dataset_stats: FilteringStep = None

    def run(
        self,
        total_corpus_bytes: int,
        max_lines: Optional[int],
    ):
        stats = self.stats
        with ExitStack() as stack:
            src_outfile = stack.enter_context(write_lines(self.src_outpath))
            trg_outfile = stack.enter_context(write_lines(self.trg_outpath))

            if max_lines:
                for line in shuffle_with_max_lines(
                    line_stream=self.yield_lines_string(stack),
                    seed=38540735095,
                    max_lines=max_lines,
                    total_byte_size=total_corpus_bytes,
                ):
                    src_line, trg_line = line.split("\t")
                    src_outfile.write(src_line)
                    trg_outfile.write(trg_line)

                stats.final_truncated.visited = stats.parallel_corpus.kept
                stats.final_truncated.kept = min(max_lines, stats.parallel_corpus.kept)
            else:
                for src_line, trg_line in self.yield_lines_tuple(stack):
                    src_outfile.write(src_line)
                    trg_outfile.write(trg_line)

                stats.final_truncated.kept = stats.parallel_corpus.kept
                stats.final_truncated.visited = stats.parallel_corpus.kept

    def yield_lines_tuple(self, stack: ExitStack) -> Generator[tuple[str, str], None, None]:
        strings_seen = WeakStringSet()
        stats = self.stats
        src_lines: Generator[str, None, None] = stack.enter_context(
            read_lines(self.datasets_src, on_enter_location=self.on_enter_location)
        )
        trg_lines: Generator[str, None, None] = stack.enter_context(
            read_lines(self.datasets_trg, on_enter_location=log_dataset)
        )

        for src_line, trg_line in zip(src_lines, trg_lines):
            # No separator is needed as the newline is included.
            line = src_line + trg_line

            if line in strings_seen:
                stats.parallel_corpus.filtered += 1
                self.dataset_stats.filtered += 1
            else:
                stats.parallel_corpus.kept += 1
                self.dataset_stats.kept += 1

                strings_seen.add(line)

                yield src_line, trg_line

    def yield_lines_string(self, stack: ExitStack) -> Generator[str, None, None]:
        for src_line, trg_line in self.yield_lines_tuple(stack):
            if "\t" in src_line or "\t" in trg_line:
                logger.error("A line contained a tab character, skipping:")
                logger.error(f" src: {src_line}")
                logger.error(f" trg: {src_line}")
            else:
                yield f"{src_line}\t{trg_line}"

    def on_enter_location(self, location):
        log_dataset(location)
        self.dataset_stats = self.stats.add_parallel_dataset(location)


def sample_corpus(
    artifacts: Path, name: str, sample_size: int, src_outpath: Path, trg_outpath: Path
):
    """
    Generate a sample of the corpus data with the following format:

    e.g.
    > cat artifacts/corpus.sample.txt
    Sentence 1 in source language
    Sentence 1 in target language

    Sentence 2 in source language
    Sentence 2 in target language

    Sentence 3 in source language
    Sentence 3 in target language
    ...
    """
    total_byte_size = src_outpath.stat().st_size + trg_outpath.stat().st_size

    with ExitStack() as stack:
        sample_path = artifacts / f"{name}.sample.txt"

        src_lines = stack.enter_context(read_lines(src_outpath))
        trg_lines = stack.enter_context(read_lines(trg_outpath))
        sample_outfile = stack.enter_context(
            write_lines(
                sample_path,
                # The browser won't know the encoding when viewing this sample without including
                # a "byte order mark", which python can do via this encoding.
                encoding="utf-8-sig",
            )
        )

        def join_src_trg():
            for src_line, trg_line in zip(src_lines, trg_lines):
                # The src and trg line each have a newline at the end. This means that
                # each sentence pair will be separate by a blank line to make for easy
                # scanning of datasets.
                yield f"{src_line}{trg_line}\n"

        logger.info("Stream in:")
        logger.info(f" - {src_outpath}")
        logger.info(f" - {trg_outpath}")
        logger.info(f"Write a {sample_size:,} line sample of the merged corpus:")
        logger.info(f" - {sample_path}")

        for line in shuffle_with_max_lines(
            line_stream=join_src_trg(),
            seed=9834523434,
            max_lines=sample_size,
            total_byte_size=total_byte_size,
        ):
            sample_outfile.write(line)


def get_datasets(src: str, trg: str, datasets_glob: str):
    dataset_paths: list[str] = glob(datasets_glob)
    datasets_src: list[Path] = []
    datasets_trg: list[Path] = []
    dataset_paths.sort()

    total_corpus_bytes = 0

    for dataset in dataset_paths:
        path = Path(dataset)
        if dataset.endswith(f"{src}.zst"):
            datasets_src.append(path)
        elif dataset.endswith(f"{trg}.zst"):
            datasets_trg.append(path)
        else:
            raise Exception(f"Dataset does not match naming scheme: {dataset}")

        formatted_size, bytes = get_human_readable_file_size(path)
        logger.info(f" - {path} ({formatted_size})")
        total_corpus_bytes += bytes

    return datasets_src, datasets_trg, total_corpus_bytes


def main() -> None:
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
        "--max_lines",
        type=str,
        default="None",
        help="The (optionally) maximum number of sentences that will be merged.",
    )

    parser.add_argument(
        "--sample_size", type=int, default=10_000, help="Generate a random sample of sentences."
    )

    parser.add_argument(
        "--artifacts",
        type=Path,
        help="The path to the artifacts directory.",
    )

    parser.add_argument(
        "--name",
        type=str,
        help='The final corpus name, e.g. "corpus" will output a "corpus.en.zst" file.',
    )

    args = parser.parse_args()

    datasets_src, datasets_trg, total_corpus_bytes = get_datasets(
        args.src, args.trg, args.datasets_glob
    )

    logger.info("Parallel datasets:")

    src_outpath = args.artifacts / f"{args.name}.{args.src}.zst"
    trg_outpath = args.artifacts / f"{args.name}.{args.trg}.zst"

    stats = FilteringStatistics(args.artifacts / args.name)

    max_lines: Optional[int] = None
    if args.max_lines != "None":
        max_lines = int(args.max_lines)

    deduplicate_corpus = DeduplicateCorpus(
        datasets_src,
        datasets_trg,
        src_outpath,
        trg_outpath,
        stats,
    )

    deduplicate_corpus.run(total_corpus_bytes, max_lines)

    sample_corpus(
        artifacts=args.artifacts,
        name=args.name,
        sample_size=args.sample_size,
        src_outpath=src_outpath,
        trg_outpath=trg_outpath,
    )

    stats.save_json()


if __name__ == "__main__":
    main()
