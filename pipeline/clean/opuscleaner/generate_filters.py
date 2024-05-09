"""
Generates filter config for a dataset based on defaults to use in OpusCleaner
"""

import argparse
import json
import os
from typing import Optional

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def find_custom_filter(src: str, trg: str, dataset: str) -> Optional[str]:
    # TODO: we'll likely need to move to a separate repo for those
    # TODO: to not include all filters for all languages in TC artifacts

    # workaround: we use "_" or "/" to separate the dataset version for OPUS datasets and OpusCleaner uses "-"
    idx = dataset.rfind("/") if "/" in dataset else dataset.rfind("_")
    dataset_opus = f"{dataset[:idx]}-{dataset[idx + 1:]}" if idx else ""

    # note: do not call the folder with default filters "filters" because it's a magic word for opuscleaner-clean
    # and it starts processing such folder

    # First look in the langauge specific directory, then in the common directory
    # as language specific dataset configs should override the generic ones.
    # The cleaning rules are symmetrical, so we can use one language specific config for both directions
    paths = [
        f"{CURRENT_FOLDER}/configs/{src}-{trg}/{dataset}.filters.json",
        f"{CURRENT_FOLDER}/configs/{src}-{trg}/{dataset_opus}.filters.json",
        f"{CURRENT_FOLDER}/configs/{trg}-{src}/{dataset}.filters.json",
        f"{CURRENT_FOLDER}/configs/{trg}-{src}/{dataset_opus}.filters.json",
        f"{CURRENT_FOLDER}/configs/{dataset}.filters.json",
        f"{CURRENT_FOLDER}/configs/{dataset_opus}.filters.json",
    ]

    for path in paths:
        if os.path.exists(path):
            return path
    return None


def build_config(config_path: str, src: str, trg: str) -> str:
    # TODO: ideally "other" for "deescape-special-chars" should be replaced to <trg> for supported languages
    with open(config_path) as f:
        config_str = f.read()
        config_str = config_str.replace("<src>", src).replace("<trg>", trg)
        # this replacement is required for the custom filters that were copied from OpusCleaner UI too
        abs_path_patterns = f"{CURRENT_FOLDER}/configs/remove_frequent_patterns.txt"
        config_str = config_str.replace("configs/remove_frequent_patterns.txt", abs_path_patterns)
        return json.loads(config_str)


def generate(dataset: str, output: str, src: str, trg: str) -> None:
    # look whether there are custom filters produced by OpusCleaner UI first
    # if a custom filter is not found, use defaults
    # filter_path = (
    #     find_custom_filter(src, trg, dataset) or f"{CURRENT_FOLDER}/configs/default.filters.json"
    # )
    filter_path = f"{CURRENT_FOLDER}/configs/default.filters.json"
    print(f"Using filter {filter_path}")
    config = build_config(filter_path, src, trg)
    with open(output, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_prefix", metavar="INPUT_PREFIX", type=str, help="Dataset file prefix"
    )
    parser.add_argument("src", metavar="SRC", type=str, help="Source language code")
    parser.add_argument("trg", metavar="TRG", type=str, help="Target language code")
    parser.add_argument("dataset", metavar="DATASET", type=str, help="Dataset name")
    parser.add_argument("output", metavar="OUTPUT_PATH", type=str, help="Write filter config here")
    args = parser.parse_args()

    generate(args.dataset, args.output, args.src, args.trg)
