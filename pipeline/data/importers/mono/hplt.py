import json
import random
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from pipeline.common.datasets import (
    CountingStep,
    FilteringStep,
    Statistics,
    WeakStringSet,
)
from pipeline.common.downloads import location_exists, read_lines, write_lines
from pipeline.common.logging import get_logger
from pipeline.common.memory import log_memory

logger = get_logger(__name__)

random.seed(38947598475)


@dataclass
class HPLTDocument:
    """
    A structured type for the HPLT document entry in a jsonl file.
    https://hplt-project.org/datasets/v1.2

    Usage:
        doc = HPLTDocument(**raw_json)
    """

    id: int
    document_lang: str
    # The list of scores for each line.
    scores: list[float]
    # The detected language for the sentence.pl
    langs: list[str]
    # All of the text, split by newlines.
    text: str
    # Where this URL was scraped: https://hip.univ-orleans.fr/ipac20/ipac.jsp?session=166R9L62F6723.6585606&profile=scd&menu=search&ts=1660916246772
    url: str
    # e.g. "cc40"
    collection: str

    def __post_init__(self):
        # The sentences in the text, which were separated by newliens.
        self.lines = self.text.split("\n")


class FilteringStatistics(Statistics):
    """
    Gather statistics about the filtering process.
    """

    def __init__(self, dataset_path: Path) -> None:
        super().__init__(dataset_path)
        self.shards = FilteringStep(
            "How many shards were sampled from. Each shard contains a subset of the "
            "total datasets available.",
        )
        self.visited_lines = FilteringStep(
            "How many lines were visited and kept from the HPLT documents.",
        )
        self.document_count = CountingStep(
            "How many documents were visited. This can help represent data diversity.",
        )
        self.duplicate_lines = CountingStep(
            "Of the collected lines, this counts how many were duplicates and discarded.",
        )
        self.final_lines = CountingStep(
            "How many lines were actually written.",
        )

    def count_shards_visited(self, *_args):
        self.shards.filtered -= 1
        self.shards.kept += 1


def language_has_hplt_support(language: str) -> bool:
    return location_exists(
        f"https://data.hplt-project.org/one/monotext/cleaned/{language}_map.txt"
    )


def load_shuffled_shard_urls(language: str) -> list[str]:
    """
    Download the list of shards, e.g.

    https://data.hplt-project.org/one/monotext/cleaned/en/en_100.jsonl.zst
    https://data.hplt-project.org/one/monotext/cleaned/en/en_101.jsonl.zst
    https://data.hplt-project.org/one/monotext/cleaned/en/en_102.jsonl.zst
    https://data.hplt-project.org/one/monotext/cleaned/en/en_103.jsonl.zst
    https://data.hplt-project.org/one/monotext/cleaned/en/en_104.jsonl.zst
    ...
    https://data.hplt-project.org/one/monotext/cleaned/en/en_110.jsonl.zst
    """

    # TODO: migrate to HPLT 2.0: https://hplt-project.org/datasets/v2.0, https://github.com/mozilla/firefox-translations-training/issues/884
    url = f"https://data.hplt-project.org/one/monotext/cleaned/{language}_map.txt"
    logger.info(f"Downloading shard list: {url}")

    with read_lines(url) as lines:
        shard_urls = []
        for line in lines:
            shard_urls.append(line.strip())
    random.Random(url).shuffle(shard_urls)

    logger.info(f"Available shards for {language}:")
    for lines in shard_urls:
        logger.info(f" - {lines}")
    return shard_urls


def download_hplt(
    language: str,
    hlpt_min_fluency: float,
    max_characters: int,
    max_lines: int,
    file_destination: Path,
):
    """
    Downloads and filters the HPLT dataset.
    https://hplt-project.org/datasets/v1.2

    In the English data with a fluency score of 0.9, about 5% of the total sentences visited
    are fluent enough to keep. That means to generate a monolingual dataset of 100 million
    lines 2 billion lines need to be visited.

    Parameters:
     - language: The BCP 47 language code to filter the documents.
     - hlpt_min_fluency: The minimum score a sentence must have to be included in the final dataset.
     - max_characters: The maximum number of characters to merge sentences in the document before writing. 0 - preserve the lines as in the dataset
     - max_lines: The maximum number of lines to include in the final dataset.
     - file_destination: The destination path where the final dataset will be written.
    """

    with ExitStack() as stack:
        stats = FilteringStatistics(file_destination)

        outfile = stack.enter_context(write_lines(file_destination))

        shuffled_shard_urls = load_shuffled_shard_urls(language)
        stats.shards.filtered = len(shuffled_shard_urls)

        # The shard URLs are shuffled, and then streamed into the read_lines iterator.
        # This iterator can work over multiple documents. The first document is loaded,
        # and then the documents in the shard are read in order from that shard. After
        # the first shard is read, the iterator continues with the next shards until
        # enough fluent sentences are collected. At this point the remaining shards
        # will not be visited.
        document_stream = stack.enter_context(
            read_lines(shuffled_shard_urls, on_enter_location=stats.count_shards_visited)
        )

        strings_seen = WeakStringSet()
        accumulated_text: str = ""
        cumulative_char_count = 0
        visited_lines = 0

        def maybe_write_accumulated_text():
            """
            Since the loop below is building up paragraphs of text, we only want to write
            out a line when enough text has been accumulated. The paragraph should be
            written out when either the text gets too long, or the next line is discarded.
            """
            nonlocal accumulated_text
            nonlocal cumulative_char_count
            cumulative_char_count = 0
            if accumulated_text:
                if accumulated_text in strings_seen:
                    stats.duplicate_lines.value += 1
                else:
                    outfile.write(accumulated_text + "\n")
                    stats.final_lines.value += 1
                    strings_seen.add(accumulated_text)
                accumulated_text = ""

        for document_json in document_stream:
            stats.document_count.value += 1

            document = HPLTDocument(**json.loads(document_json))

            maybe_write_accumulated_text()

            # Visit the lines in the document.
            for score, lang_item, line in zip(document.scores, document.langs, document.lines):
                visited_lines += 1

                # Check for the fluency scores.
                if lang_item == language and score >= hlpt_min_fluency:
                    char_count = len(line)

                    if char_count > max_characters:
                        # This segment is too long, or we don't merge document lines (max_characters is 0)
                        if max_characters == 0:
                            accumulated_text = line
                        maybe_write_accumulated_text()
                    else:
                        stats.visited_lines.kept += 1

                        # Determine if this sentence should be added to the previous one or
                        # written out as a new line. Only concurrent sentences that meet
                        # the fluency requirement will be combined together.
                        if cumulative_char_count + char_count > max_characters:
                            # This line would be too long, write it out.
                            maybe_write_accumulated_text()

                        cumulative_char_count += char_count
                        # Collect this line to write.
                        if accumulated_text:
                            accumulated_text = f"{accumulated_text} {line}"
                        else:
                            accumulated_text = line
                else:
                    maybe_write_accumulated_text()

                if visited_lines % 5_000_000 == 0:
                    logger.info(f"Visited {visited_lines:,} lines")
                    logger.info(f"Kept {stats.visited_lines.kept:,}.")
                    logger.info(f"Wrote {stats.final_lines.value:,} out of {max_lines:,}.")
                    log_memory()

                if stats.final_lines.value == max_lines:
                    break

            if stats.final_lines.value == max_lines:
                break
            maybe_write_accumulated_text()

        stats.visited_lines.filtered = visited_lines - stats.visited_lines.kept
        logger.info(f"Wrote {stats.final_lines.value:,} lines to: {file_destination}")
        stat_path = stats.save_json()
        logger.info(f"Saved filtering stats to: {stat_path}")
