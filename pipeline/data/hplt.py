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
        # The sentences in the text, which were separated by newlines.
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
        self.filtered_doc_locale = CountingStep(
            "How many lines were filtered based on document locale.",
        )
        self.filtered_line_locale = CountingStep(
            "How many lines were filtered based on line locales.",
        )
        self.filtered_doc_score = CountingStep(
            "How many lines were filtered based on document scores.",
        )
        self.filtered_too_long = CountingStep(
            "How many lines were filtered based on length.",
        )

    def count_shards_visited(self, *_args):
        self.shards.filtered -= 1
        self.shards.kept += 1


def get_hplt_locale(lang_iso6931: str) -> str:
    """
    Converts language in ISO-693-1 format to the HPLT format.
    For example, ru -> rus_Cyrl
    """
    # icu return Kore by default which is a mix of Hang and Hani
    if lang_iso6931 == "ko":
        return "kor_Hang"
    if lang_iso6931 == "zh":
        return "cmn_Hant"
    locale = icu.Locale(lang_iso6931)
    # add default script
    locale = icu.Locale.addLikelySubtags(locale)
    hplt_locale = f"{locale.getISO3Language()}_{locale.getScript()}"
    return hplt_locale


def get_hplt_map_url(hplt_locale: str) -> str:
    return f"https://data.hplt-project.org/three/sorted/{hplt_locale}.map"


def language_has_hplt_support(language: str) -> bool:
    hplt_locale = get_hplt_locale(language)
    hplt_map = get_hplt_map_url(hplt_locale)
    return location_exists(hplt_map)


def load_shuffled_shard_urls(hplt_locale: str, min_doc_score: float) -> list[str]:
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
            # extract doc score and filter
            # 8 is document score here
            # https://data.hplt-project.org/three/sorted/cmn_Hant/8_1.jsonl.zst
            doc_score = int(line.split("/")[-1].split("_")[0])
            if doc_score >= min_doc_score:
                shard_urls.append(line.strip())
    random.Random(url).shuffle(shard_urls)

    logger.info(f"Available shards for {hplt_locale}:")
    for lines in shard_urls:
        logger.info(f" - {lines}")
    return shard_urls


class HpltDownloader:
    """
    Downloads and filters the HPLT dataset.
    https://hplt-project.org/datasets/v2.0

    Parameters:
     - language: The BCP 47 language code to filter the documents.
     - hplt_min_doc_score: The minimum score a document must have to be included in the final dataset.
     - max_characters: The maximum number of characters to merge sentences in the document before writing if enabled.
                       Also filters lines that are too long.
     - max_lines: The maximum number of lines to include in the final dataset.
     - file_destination: The destination path where the final dataset will be written.
     - merge_lines: Whether to accumulate line of the same document in one segment until max_characters is reached.
    """

    def __init__(
        self,
        language: str,
        hplt_min_doc_score: float,
        max_characters: int,
        max_lines: int,
        file_destination: Path,
        merge_lines: bool,
    ) -> None:
        self.merge_lines = merge_lines
        self.max_lines = max_lines
        self.max_characters = max_characters
        self.hplt_min_doc_score = hplt_min_doc_score
        self.hplt_locale = get_hplt_locale(language)
        self.accumulated_text = ""
        self.cumulative_char_count = 0
        self.visited_lines = 0
        self.file_destination = file_destination
        self.stats = FilteringStatistics(file_destination)
        self.strings_seen = WeakStringSet()
        self.stack = ExitStack()
        self.outfile = self.stack.enter_context(write_lines(file_destination))

    def close(self):
        self.stack.close()

    def download(self):
        try:
            self._run_download()
        finally:
            self.close()

    def _run_download(self):
        logger.info(f"Using HPLT locale {self.hplt_locale}")
        shuffled_shard_urls = load_shuffled_shard_urls(self.hplt_locale, self.hplt_min_doc_score)
        self.stats.shards.filtered = len(shuffled_shard_urls)

        # The shard URLs are shuffled, and then streamed into the read_lines iterator.
        # This iterator can work over multiple documents. The first document is loaded,
        # and then the documents in the shard are read in order from that shard. After
        # the first shard is read, the iterator continues with the next shards until
        # enough fluent sentences are collected. At this point the remaining shards
        # will not be visited.
        document_stream = self.stack.enter_context(
            read_lines(shuffled_shard_urls, on_enter_location=self.stats.count_shards_visited)
        )

        for document_json in document_stream:
            self.stats.document_count.value += 1
            document = HPLTDocument(**json.loads(document_json))
            overall_doc_score = document.doc_scores[0]
            doc_lang = document.lang[0]

            self._maybe_write_accumulated_text()

            # HPLT 2.0 uses document level scores
            if overall_doc_score < self.hplt_min_doc_score:
                self.stats.filtered_doc_score.value += 1
                continue

            # We want only documents written primarily in the target language
            if doc_lang != self.hplt_locale:
                self.stats.filtered_doc_locale.value += 1
                continue

            # Visit the lines in the document.
            for line_locale, line in zip(document.seg_langs, document.lines):
                self.visited_lines += 1
                self._process_line(line_locale, line)
                if self.visited_lines % 5_000_000 == 0:
                    logger.info(f"Visited {self.visited_lines:,} lines")
                    logger.info(f"Kept {self.stats.visited_lines.kept:,}.")
                    logger.info(
                        f"Wrote {self.stats.final_lines.value:,} out of {self.max_lines:,}."
                    )
                    log_memory()

                if self.stats.final_lines.value == self.max_lines:
                    break

            if self.stats.final_lines.value == self.max_lines:
                break

            self._maybe_write_accumulated_text()

        self.stats.visited_lines.filtered = self.visited_lines - self.stats.visited_lines.kept
        logger.info(f"Wrote {self.stats.final_lines.value:,} lines to: {self.file_destination}")
        stat_path = self.stats.save_json()
        logger.info(f"Saved filtering stats to: {stat_path}")

    def _process_line(self, line_locale: str, line: str):
        # Line locale does not match expected locale, filter
        if line_locale != self.hplt_locale:
            self.stats.filtered_line_locale.value += 1
            self._maybe_write_accumulated_text()
            return

        char_count = len(line)
        # Filter long segments
        if char_count > self.max_characters:
            self.stats.filtered_too_long.value += 1
            self._maybe_write_accumulated_text()
            return

        # Just write the current line if merging is disabled
        if not self.merge_lines:
            self.accumulated_text = line
            self.stats.visited_lines.kept += 1
            self._maybe_write_accumulated_text()
            return

        # Text accumulation mode starts here

        self.stats.visited_lines.kept += 1

        # Determine if this sentence should be added to the previous one or
        # written out as a new line.
        if self.cumulative_char_count + char_count + 1 > self.max_characters:
            # This line would be too long, write it out.
            self._maybe_write_accumulated_text()

        self.cumulative_char_count += char_count
        # Collect this line to write.
        if self.accumulated_text:
            self.accumulated_text = f"{self.accumulated_text} {line}"
            # count the whitespace
            self.cumulative_char_count += 1
        else:
            self.accumulated_text = line

    def _maybe_write_accumulated_text(self):
        """
        Since the loop below is building up paragraphs of text, we only want to write
        out a line when enough text has been accumulated. The paragraph should be
        written out when either the text gets too long, or the next line is discarded.
        """

        self.cumulative_char_count = 0
        if self.accumulated_text:
            if self.accumulated_text in self.strings_seen:
                self.stats.duplicate_lines.value += 1
            else:
                self.outfile.write(self.accumulated_text + "\n")
                self.stats.final_lines.value += 1
                self.strings_seen.add(self.accumulated_text)
            self.accumulated_text = ""
