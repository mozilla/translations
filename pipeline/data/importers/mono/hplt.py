import json
import random
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import icu

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
    https://hplt-project.org/datasets/v2.0
    """

    def __init__(self, **json):
        self.lang = json["lang"]
        self.doc_scores = json["doc_scores"]
        self.seg_langs = json["seg_langs"]
        # The sentences in the text, which were separated by newliens.
        self.lines = json["text"].split("\n")

    # The list of detected document languages where the first language is most probable.
    # For example: [zho_Hans, zho_Hant, eng_Latn]
    lang: list[str]
    # The list of document scores from web-docs-scorer where the first score is the overall document score (WDS_score) followed by 8 subscores.
    # All the scores are from 0 to 10.
    # See https://github.com/pablop16n/web-docs-scorer/
    # For example, [8.3, 10, 10, 9.9, 10, 10, 10, 4, 0]
    doc_scores: list[float]
    # The detected language for each line (segment).
    # For example: [yue_Hant, zho_Hans, zho_Hans, zho_Hant, unk, ... ]
    seg_langs: list[str]
    # All of the text, split by newlines.
    lines: list[str]


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


def get_hplt_locale(lang_iso6931: str) -> str:
    """
    Converts language in ISO-693-1 format to the HPLT format.
    For example, ru -> rus_Cyrl
    """
    locale = icu.Locale(lang_iso6931)
    # add default script
    locale = icu.Locale.addLikelySubtags(locale)
    hplt_locale = f"{locale.getISO3Language()}_{locale.getScript()}"
    return hplt_locale


def get_hplt_map_url(hplt_locale: str) -> str:
    return f"https://data.hplt-project.org/two/cleaned/{hplt_locale}_map.txt"


def language_has_hplt_support(language: str) -> bool:
    hplt_locale = get_hplt_locale(language)
    hplt_map = get_hplt_map_url(hplt_locale)
    return location_exists(hplt_map)


def load_shuffled_shard_urls(hplt_locale: str) -> list[str]:
    """
    Download the list of shards, e.g.
    https://data.hplt-project.org/two/cleaned/rus_Cyrl/1.jsonl.zst
    https://data.hplt-project.org/two/cleaned/rus_Cyrl/2.jsonl.zst
    ...
    https://data.hplt-project.org/two/cleaned/rus_Cyrl/10.jsonl.zst
    """

    url = get_hplt_map_url(hplt_locale)
    logger.info(f"Downloading shard list: {url}")

    with read_lines(url) as lines:
        shard_urls = []
        for line in lines:
            shard_urls.append(line.strip())
    random.Random(url).shuffle(shard_urls)

    logger.info(f"Available shards for {hplt_locale}:")
    for lines in shard_urls:
        logger.info(f" - {lines}")
    return shard_urls


def download_hplt(
    language: str,
    hlpt_min_doc_score: float,
    max_characters: int,
    max_lines: int,
    file_destination: Path,
):
    """
    Downloads and filters the HPLT dataset.
    https://hplt-project.org/datasets/v2.0

    Parameters:
     - language: The BCP 47 language code to filter the documents.
     - hlpt_min_doc_score: The minimum score a document must have to be included in the final dataset.
     - max_characters: The maximum number of characters to merge sentences in the document before writing. 0 - preserve the lines as in the dataset
     - max_lines: The maximum number of lines to include in the final dataset.
     - file_destination: The destination path where the final dataset will be written.
    """

    with ExitStack() as stack:
        stats = FilteringStatistics(file_destination)
        hplt_locale = get_hplt_locale(language)
        logger.info(f"Using HPLT locale {hplt_locale}")

        outfile = stack.enter_context(write_lines(file_destination))

        shuffled_shard_urls = load_shuffled_shard_urls(hplt_locale)
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

            overall_doc_score = document.doc_scores[0]
            doc_lang = document.lang[0]

            # HPLT 2.0 uses document level scores
            # We want only documents written primarily in the target language
            if overall_doc_score < hlpt_min_doc_score or doc_lang != hplt_locale:
                continue

            # Visit the lines in the document.
            for lang_item, line in zip(document.seg_langs, document.lines):
                visited_lines += 1

                if lang_item == hplt_locale:
                    char_count = len(line)

                    if char_count > max_characters:
                        # This segment is too long, or we don't merge document lines (max_characters is 0)
                        if max_characters == 0:
                            accumulated_text = line
                        maybe_write_accumulated_text()
                    else:
                        stats.visited_lines.kept += 1

                        # Determine if this sentence should be added to the previous one or
                        # written out as a new line.
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
