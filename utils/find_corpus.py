#!/usr/bin/env python3
"""
Finds all opus datasets for a language pair and prints them to set config settings.

Usage:
    task find-corpus -- en ca
    task find-corpus -- en fr --importer opus
"""

import argparse
import logging
import re
import sys
from typing import Any, Iterable, Literal, NamedTuple, Optional, TypeVar, Union

import humanize
import requests

from pipeline.data.hplt import language_has_hplt_support
from pipeline.langs.codes import LangCode


class OpusDataset(NamedTuple):
    # The name of this dataset, e.g. "CCAligned"
    corpus: str
    # This is a blank string at the time of this writing.
    documents: str

    # 'moses'
    preprocessing: str
    # The language tag.
    source: str
    # The language tag.
    target: str
    # The URL to the download
    url: str
    # For example "v1"
    version: str

    alignment_pairs: int
    id: int
    # Size in KiB
    size: int
    source_tokens: int
    target_tokens: int

    latest: Union[Literal["True"], Literal["False"]]

    def corpus_key(self) -> str:
        return f"opus_{self.corpus}/{self.version}"

    def website_url(self) -> str:
        return f"https://opus.nlpl.eu/{self.corpus}/{self.source}&{self.target}/{self.version}/{self.corpus}"

    def humanize_size(self) -> str:
        return humanize.naturalsize(self.size * 1024)


def fetch_opus(source: LangCode, target: LangCode) -> list[OpusDataset]:
    # This API is documented: https://opus.nlpl.eu/opusapi/
    url = f"https://opus.nlpl.eu/opusapi/?source={source.opus()}&target={target.opus()}&preprocessing=moses&version=latest"

    datasets = requests.get(url).json()

    # Convert the response into a typed object that is sorted.
    datasets_typed = [OpusDataset(**corpus_data) for corpus_data in datasets.get("corpora", [])]
    return sorted(datasets_typed, key=lambda x: x.alignment_pairs or 0, reverse=True)


def get_opus(source: LangCode, target: LangCode, download_url: bool):
    print("")
    print("┌──────────────────────────────────────┐")
    print(f"│ OPUS ({source.opus()}-{target.opus()}) - https://opus.nlpl.eu/ │")
    print("└──────────────────────────────────────┘")

    datasets = fetch_opus(source, target)

    print_table(
        [
            [
                "Dataset",
                "Code",
                "Sentences",
                "Size",
                "URL",
            ],
            *[
                [
                    dataset.corpus,
                    dataset.corpus_key(),
                    dataset.alignment_pairs,
                    dataset.humanize_size(),
                    dataset.url if download_url else dataset.website_url(),
                ]
                for dataset in datasets
                if dataset.alignment_pairs
            ],
        ]
    )

    names = [dataset.corpus_key() for dataset in datasets]
    print_yaml(names, exclude=["OPUS100v", "WMT-News"])


def fetch_sacrebleu(source: LangCode, target: LangCode) -> dict[str, Any]:
    import sacrebleu

    src_sb = source.sacrebleu()
    trg_sb = target.sacrebleu()

    return {
        name: entry
        for name, entry in sacrebleu.DATASETS.items()
        if f"{src_sb}-{trg_sb}" in entry.langpairs or f"{trg_sb}-{src_sb}" in entry.langpairs
    }


def get_sacrebleu(source: LangCode, target: LangCode):
    datasets_dict = fetch_sacrebleu(source, target)

    print("")
    print("┌─────────────────────────────────────────────────────────┐")
    print(
        f"│ sacrebleu ({source.sacrebleu()}-{target.sacrebleu()}) - https://github.com/mjpost/sacrebleu │"
    )
    print("└─────────────────────────────────────────────────────────┘")
    print_table(
        [
            ["Dataset", "Description", "URLs"],
            *[
                [
                    #
                    name,
                    dataset.description,
                    ", ".join(dataset.data),
                ]
                for name, dataset in datasets_dict.items()
            ],
        ]
    )
    print_yaml(list(f"sacrebleu_{name}" for name in datasets_dict.keys()))


def get_size(tags: list[str]) -> str:
    size = next(
        filter(
            lambda tag: tag.startswith("size_categories:"),
            tags,
        ),
        None,
    )

    if not size or size == "unknown":
        return ""

    # Lowercase the text since it's not consistent.
    return size.replace("size_categories:", "").lower()


def get_language_count(tags: list[str]):
    count = 0
    for tag in tags:
        if tag.startswith("language:"):
            count = count + 1
    return count


HF_DATASET_SIZES = {
    "": 0,
    "unknown": 0,
    "n<1k": 1,
    "1k<n<10k": 2,
    "10k<100k": 3,
    "10k<n<100k": 3,
    "100k<n<1m": 4,
    "1m<n<10m": 5,
    "10m<n<100m": 6,
    "100m<n<1b": 7,
    "1b<n<10b": 8,
    "10b<n<100b": 9,
    "100b<n<1t": 10,
}


def get_huggingface_monolingual(language: LangCode):
    """
    Returns monolingual datasets ordered by size. Datasets with few downloads are ignored
    as they are probably low quality and not trustworthy.
    """
    from huggingface_hub import DatasetFilter, HfApi

    api = HfApi()
    lang_hf = language.huggingface()

    datasets = list(
        api.list_datasets(
            filter=DatasetFilter(
                #
                language=lang_hf,
                multilinguality="monolingual",
            )
        )
    )
    datasets.sort(key=lambda dataset: -dataset.downloads)
    datasets.sort(key=lambda dataset: -HF_DATASET_SIZES.get(get_size(dataset.tags), 0))

    print("")
    print("┌─────────────────────────────────────────────────┐")
    print(f"│ huggingface monolingual data ({lang_hf})               │")
    print("└─────────────────────────────────────────────────┘")
    print_table(
        [
            ["ID", "Size", "Downloads"],
            *[
                [
                    #
                    f"https://huggingface.co/datasets/{dataset.id}",
                    get_size(dataset.tags),
                    dataset.downloads,
                ]
                for dataset in datasets
                if is_useful_dataset(dataset)
            ],
        ]
    )


def get_huggingface_parallel(source: LangCode, target: LangCode):
    """
    Returns parallel datasets ordered by size. Datasets with few downloads are ignored
    as they are probably low quality and not trustworthy.
    """
    from huggingface_hub import DatasetFilter, HfApi

    src_hf = source.huggingface()
    trg_hf = target.huggingface()

    api = HfApi()

    datasets = list(
        api.list_datasets(
            filter=DatasetFilter(
                #
                language=[src_hf, trg_hf],
            )
        )
    )
    datasets.sort(key=lambda dataset: -dataset.downloads)
    datasets.sort(key=lambda dataset: -HF_DATASET_SIZES.get(get_size(dataset.tags), 0))

    print("")
    print(
        "┌────────────────────────────────────────────────────────────────────────────────────────────────────┐"
    )
    print(
        f"│ huggingface parallel data https://huggingface.co/datasets?language=language:{src_hf},language:{trg_hf}"
    )
    print(
        "└────────────────────────────────────────────────────────────────────────────────────────────────────┘"
    )
    print_table(
        [
            ["ID", "Size", "Downloads"],
            *[
                [
                    #
                    f"https://huggingface.co/datasets/{dataset.id}",
                    get_size(dataset.tags),
                    dataset.downloads,
                ]
                for dataset in datasets
                if is_useful_dataset(dataset)
            ],
        ]
    )


def is_useful_dataset(dataset: Any) -> bool:
    """Determines if a dataset is useful or not."""
    return "task_categories:automatic-speech-recognition" not in dataset.tags


def get_huggingface_any(language: LangCode):
    """
    Returns parallel datasets ordered by size. Datasets with few downloads are ignored
    as they are probably low quality and not trustworthy.
    """
    from huggingface_hub import DatasetFilter, HfApi

    api = HfApi()
    lang_hf = language.huggingface()

    datasets = list(
        api.list_datasets(
            filter=DatasetFilter(
                #
                language=lang_hf,
            )
        )
    )

    datasets.sort(key=lambda dataset: -dataset.downloads)
    datasets.sort(key=lambda dataset: -HF_DATASET_SIZES.get(get_size(dataset.tags), 0))

    print("")
    print("┌─────────────────────────────────────────────────────────────────────────────┐")
    print(f"│ huggingface any data https://huggingface.co/datasets?language=language:{lang_hf}")
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    print_table(
        [
            ["ID", "Size", "Downloads"],
            *[
                [
                    #
                    f"https://huggingface.co/datasets/{dataset.id}",
                    get_size(dataset.tags),
                    dataset.downloads,
                ]
                for dataset in datasets
                if is_useful_dataset(dataset)
            ],
        ]
    )


def get_remote_file_size(
    url: str, display_not_200: bool = True
) -> tuple[Optional[int], Optional[str]]:
    try:
        response = requests.head(url, timeout=1, allow_redirects=True)

        if response.ok:
            if "Content-Length" in response.headers:
                int_size = int(response.headers.get("Content-Length", 0))
                return int_size, humanize.naturalsize(int_size)
            # Try again using GET.
        else:
            if display_not_200:
                print(f"Failed to retrieve file information for: {url}")
                print(f"Status code: {response.status_code}")
            return None, None

        # Sometimes when the HEAD does not have the Content-Length, the GET response does.
        response = requests.get(url, timeout=1, allow_redirects=True, stream=True)
        int_size = int(response.headers.get("Content-Length", 0))
        response.close()
        return int_size, humanize.naturalsize(int_size)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None


T = TypeVar("T")

from mtdata.entry import Entry


def fetch_mtdata(source: LangCode, target: LangCode) -> dict[str, Entry]:
    """
    Returns a dict that maps the corpus key to the mtdata entry.
    """
    # mtdata outputs debug logs
    logging.disable(logging.CRITICAL)

    from mtdata.entry import BCP47Tag
    from mtdata.index import get_entries

    source_tricode = source.mtdata()
    target_tricode = target.mtdata()
    entries = sorted(
        get_entries((BCP47Tag(source_tricode), BCP47Tag(target_tricode)), None, None, True),
        key=lambda entry: entry.did.group,
    )

    def get_corpus_key(entry):
        return (
            f"mtdata_{entry.did.group}-{entry.did.name}-{entry.did.version}-{entry.did.lang_str}"
        )

    entries = {get_corpus_key(entry): entry for entry in entries}

    excludes = ["opus", "newstest", "unv1", "elrc"]  # lowercase excludes.

    def is_excluded(corpus_key: str) -> bool:
        for exclude in excludes:
            if exclude in corpus_key.lower():
                return True
        return False

    # Filter out the excluded entries.
    return {
        corpus_key: entry for corpus_key, entry in entries.items() if not is_excluded(corpus_key)
    }


def get_mtdata(source: LangCode, target: LangCode):
    entries = fetch_mtdata(source, target)

    print("")
    print("┌──────────────────────────────────────────────────────────────┐")
    print(
        f"│ mtdata ({source.mtdata()}-{target.mtdata()}) - https://github.com/thammegowda/mtdata     │"
    )
    print("└──────────────────────────────────────────────────────────────┘")
    print_table(
        [
            [
                "Dataset",
                "URL",
                # "Size",
            ],
            *[
                [
                    #
                    corpus_key,
                    entry.url,
                    # get_remote_file_size(entry.url),
                ]
                for corpus_key, entry in entries.items()
                # Filter out the excludes
            ],
        ]
    )

    print_yaml(entries.keys())


class MonoDataset(NamedTuple):
    name: str
    url: str
    size: Optional[int]
    display_size: Optional[str]
    lines_num: Optional[int]


def fetch_news_crawl(lang: LangCode) -> list[MonoDataset]:
    lang_nc = lang.newscrawl()

    base_url = f"https://data.statmt.org/news-crawl/{lang_nc}/"
    response = requests.get(base_url, allow_redirects=True)

    datasets = []
    if response.ok:
        # Example row: (indentation and newlines added)
        # <tr>
        #     <td valign="top"><img src="/icons/compressed.gif" alt="[   ]"></td>
        #     <td><a href="news.2013.en.shuffled.deduped.gz">news.2013.en.shuffled.deduped.gz</a></td>
        #     <td align="right">2019-01-14 10:23  </td>
        #     <td align="right">1.2G</td>
        #     <td>&nbsp;</td>
        # </tr>

        regex = re.compile(
            r"""
            # Match the file name year.
            # >news.2008.en.shuffled.deduped.gz<
            #       ^^^^
            >news.(\d+)\.\w+\.shuffled\.deduped\.gz<
            [^\n]*

            # Match the file size and unit.
            # <td align="right">176M</td>
            #                   ^^^^
            <td\ align="right">
                ([\d\.]+)(\w+)
            </td>
        """,
            re.VERBOSE,
        )

        matches = re.findall(regex, response.text)

        if matches:
            for year, size_number, size_unit in matches:
                multiplier = 1
                if size_unit == "K":
                    multiplier = 1_000
                elif size_unit == "M":
                    multiplier = 1_000_000
                elif size_unit == "G":
                    multiplier = 1_000_000_000

                name = f"news-crawl_news.{year}"
                url = f"https://data.statmt.org/news-crawl/{lang_nc}/news.{year}.{lang_nc}.shuffled.deduped.gz"
                size = int(float(size_number) * multiplier)

                datasets.append(MonoDataset(name, url, size, f"{size_number}{size_unit}", None))
        else:
            print("The regex could not find newscrawl datasets for", lang)
    else:
        print("No newscrawl data was available for", lang)
    return datasets


def get_news_crawl(source: LangCode, target: LangCode):
    for lang in (source, target):
        datasets = fetch_news_crawl(lang)

        print("")
        print("┌─────────────────────────────────────────────────────────────────────┐")
        print(
            f"│ news-crawl ({lang.newscrawl()}) - https://github.com/data.statmt.org/news-crawl     │"
        )
        print("└─────────────────────────────────────────────────────────────────────┘")
        print_table(
            [
                [
                    "Dataset",
                    "URL",
                    "Size",
                ],
                *[[name, url, display_size] for name, url, _, display_size, _ in datasets],
            ]
        )

        print_yaml([name for name, _, _, _, _ in datasets])


def fetch_hplt(lang: LangCode) -> list[MonoDataset]:
    if language_has_hplt_support(lang):
        return [MonoDataset("hplt_mono/v3.0", "", None, None, 0)]

    return []


def get_hplt_mono(source: LangCode, target: LangCode):
    for lang in (source, target):
        datasets = fetch_hplt(lang)

        print("")
        print("┌─────────────────────────────────────────────────────────────────────┐")
        print(f"│ hplt-mono ({lang.hplt()}) - https://data.hplt-project.org/three/sorted/  │")
        print("└─────────────────────────────────────────────────────────────────────┘")
        print_table(
            [
                [
                    "Dataset",
                    "URL",
                    "Size",
                ],
                *[[name, url, display_size] for name, url, _, display_size, _ in datasets],
            ]
        )

        print_yaml([name for name, _, _, _, _ in datasets])


def print_yaml(names: Iterable[str], exclude: list[str] = []):
    cleaned = set()
    for name in names:
        filter = False
        for ex in exclude:
            if ex.lower() in name.lower():
                filter = True
                break
        if not filter:
            cleaned.add(name)

    print("\nYAML:")
    if len(cleaned) == 0:
        print("(no datasets)\n")
    else:
        print("\n".join(sorted([f"    - {name}" for name in cleaned])))


def print_table(table: list[list[Any]]):
    """
    Nicely print a table, the first row is the header
    """

    # Compute the column lengths.
    transposed_table = list(map(list, zip(*table)))
    column_lengths = [max(len(str(x)) for x in column) for column in transposed_table]

    print("")
    for index, row in enumerate(table):
        # Print the row.
        for datum, max_len in zip(row, column_lengths):
            print(str(datum).ljust(max_len), end=" ")
        print("")

        # Print a separator between the header and the rest of the table.
        if index == 0:
            for length in column_lengths:
                print("".ljust(length, "─"), end=" ")
            print("")

    if len(table) == 1:
        print("(no datasets)")


def main(args_list: Optional[list[str]] = None) -> None:
    importers = [
        "opus",
        "sacrebleu",
        "mtdata",
        "huggingface_mono",
        "huggingface_parallel",
        "huggingface_any",
        "news-crawl",
        "hplt-mono",
    ]
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves whitespace in the help text.
    )
    parser.add_argument("source", type=str, nargs="?", help="Source language code")
    parser.add_argument("target", type=str, nargs="?", help="Target language code")
    parser.add_argument(
        "--importer",
        type=str,
        help=f"The importer to use: {', '.join(importers)}",
    )
    parser.add_argument(
        "--download_url",
        action="store_true",
        default=False,
        help="Show the download url if available.",
    )

    args = parser.parse_args(args_list)

    if not args.source or not args.target:
        parser.print_help()
        sys.exit(1)

    src = LangCode(args.source)
    trg = LangCode(args.target)

    if args.importer and args.importer not in importers:
        print(f'"{args.importer}" is not a valid importer.')
        sys.exit(1)

    if args.importer == "opus" or not args.importer:
        get_opus(src, trg, args.download_url)

    if args.importer == "sacrebleu" or not args.importer:
        get_sacrebleu(src, trg)

    if args.importer == "mtdata" or not args.importer:
        get_mtdata(src, trg)

    if args.importer == "huggingface_mono" or not args.importer:
        get_huggingface_monolingual(trg if src == "en" else src)

    if args.importer == "huggingface_parallel" or not args.importer:
        get_huggingface_parallel(src, trg)

    if args.importer == "huggingface_any" or not args.importer:
        get_huggingface_any(trg if src == "en" else src)

    if args.importer == "news-crawl" or not args.importer:
        get_news_crawl(src, trg)

    if args.importer == "hplt-mono" or not args.importer:
        get_hplt_mono(src, trg)


if __name__ == "__main__":
    main()
