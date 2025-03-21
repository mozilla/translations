#!/usr/bin/env python3
"""
Downloads a dataset and runs augmentation if needed

Example:
    python pipeline/data/dataset_importer.py \
        --type=corpus \
        --dataset=sacrebleu_aug-mix_wmt19 \
        --output_prefix=$(pwd)/test_data/augtest \
        --src=ru \
        --trg=en
"""

import argparse
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from opustrainer.modifiers.noise import NoiseModifier
from opustrainer.modifiers.placeholders import PlaceholderTagModifier
from opustrainer.modifiers.surface import TitleCaseModifier, UpperCaseModifier
from opustrainer.modifiers.typos import TypoModifier
from opustrainer.types import Modifier

from pipeline.common.downloads import compress_file, decompress_file
from pipeline.data.cjk import ChineseConverter, ChineseType

random.seed(1111)


class CompositeModifier:
    """
    Composite modifier runs several modifiers one after another
    """

    def __init__(self, modifiers: List[Modifier]):
        self._modifiers = modifiers

    def __call__(self, batch: List[str]) -> Iterable[str]:
        for mod in self._modifiers:
            batch = list(mod(batch))

        return batch


MIX_PROB = 0.05  # 5% will be augmented in the mix
PROB_1 = 1.0  # 100% chance
PROB_0 = 0.0  # 0% chance
# use lower probabilities than 1 to add inline noise into the mix
# probability 1 adds way too much noise to a corpus
NOISE_PROB = 0.05
NOISE_MIX_PROB = 0.01


def get_typos_probs() -> Dict[str, float]:
    # select 4 random types of typos
    typos = set(random.sample(list(TypoModifier.modifiers.keys()), k=4))
    # set probability 1 for selected typos and 0 for the rest
    probs = {typo: PROB_1 if typo in typos else PROB_0 for typo in TypoModifier.modifiers.keys()}
    return probs


# See documentation for the modifiers in https://github.com/mozilla/translations/blob/main/docs/training/opus-trainer.md#supported-modifiers
modifier_map = {
    "aug-typos": lambda: TypoModifier(PROB_1, **get_typos_probs()),
    "aug-title": lambda: TitleCaseModifier(PROB_1),
    "aug-upper": lambda: UpperCaseModifier(PROB_1),
    "aug-noise": lambda: NoiseModifier(PROB_1),
    "aug-inline-noise": lambda: PlaceholderTagModifier(NOISE_PROB, augment=1),
    "aug-mix": lambda: CompositeModifier(
        [
            TypoModifier(MIX_PROB, **get_typos_probs()),
            TitleCaseModifier(MIX_PROB),
            UpperCaseModifier(MIX_PROB),
            NoiseModifier(MIX_PROB),
            PlaceholderTagModifier(NOISE_MIX_PROB, augment=1),
        ]
    ),
    "aug-mix-cjk": lambda: CompositeModifier(
        [
            NoiseModifier(MIX_PROB),
            PlaceholderTagModifier(NOISE_MIX_PROB, augment=1),
        ]
    ),
    "aug-mix-nocase": lambda: CompositeModifier(
        [
            TypoModifier(MIX_PROB, **get_typos_probs()),
            NoiseModifier(MIX_PROB),
            PlaceholderTagModifier(NOISE_MIX_PROB, augment=1),
        ]
    ),
}


def run_cmd(cmd: List[str], env: Dict[str, str]):
    result = None
    # make sure to preserve the current process env vars
    env_vars = dict(os.environ)
    env_vars.update(env)
    try:
        result = subprocess.run(
            cmd,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            env=env_vars,
        )
        result.check_returncode()
    except:
        if result:
            print(result.stdout)
        raise

    print(result.stdout)


def add_alignments(corpus: List[str]) -> List[str]:
    from simalign import SentenceAligner

    # We use unsupervised aligner here because statistical tools like fast_align require a large corpus to train on
    # This is slow without a GPU and is meant to operate only on small evaluation datasets

    # Use BERT with subwords and itermax as it has a higher recall and matches more words than other methods
    # See more details in the paper: https://arxiv.org/pdf/2004.08728.pdf
    # and in the source code: https://github.com/cisnlp/simalign/blob/master/simalign/simalign.py
    # This will download a 700Mb BERT model from Hugging Face and cache it
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="i")

    alignments = []
    for line in corpus:
        src_sent, trg_sent = line.split("\t")
        sent_aln = aligner.get_word_aligns(src_sent, trg_sent)["itermax"]
        aln_str = " ".join(f"{src_pos}-{trg_pos}" for src_pos, trg_pos in sent_aln)
        alignments.append(aln_str)

    corpus_tsv = [f"{sents}\t{aln}" for sents, aln in zip(corpus, alignments)]
    return corpus_tsv


# we plan to use it only for small evaluation datasets
def augment(output_prefix: str, aug_modifer: str, src: str, trg: str):
    """
    Augment corpus on disk using the OpusTrainer modifier
    """
    if aug_modifer not in modifier_map:
        raise ValueError(f"Invalid modifier {aug_modifer}. Allowed values: {modifier_map.keys()}")

    # file paths for compressed and uncompressed corpus
    uncompressed_src = f"{output_prefix}.{src}"
    uncompressed_trg = f"{output_prefix}.{trg}"
    compressed_src = f"{output_prefix}.{src}.zst"
    compressed_trg = f"{output_prefix}.{trg}.zst"

    corpus = read_corpus_tsv(compressed_src, compressed_trg, uncompressed_src, uncompressed_trg)

    if aug_modifer in ("aug-mix", "aug-inline-noise", "aug-mix-cjk", "aug-mix-nocase"):
        # add alignments for inline noise
        # Tags modifier will remove them after processing
        corpus = add_alignments(corpus)

    modified = []
    for line in corpus:
        # recreate modifier for each line to apply randomization (for typos)
        modifier = modifier_map[aug_modifer]()
        modified += modifier([line])
    write_modified(modified, uncompressed_src, uncompressed_trg)


def read_corpus_tsv(
    compressed_src: str, compressed_trg: str, uncompressed_src: str, uncompressed_trg: str
) -> List[str]:
    """
    Decompress corpus and read to TSV
    """
    if os.path.isfile(uncompressed_src):
        os.remove(uncompressed_src)
    if os.path.isfile(uncompressed_trg):
        os.remove(uncompressed_trg)

    # Decompress the original corpus.
    decompress_file(compressed_src, keep_original=False)
    decompress_file(compressed_trg, keep_original=False)

    # Since this is only used on small evaluation sets, it's fine to load the entire dataset
    # and augmentation into memory rather than streaming it.
    with open(uncompressed_src) as f:
        corpus_src = [line.rstrip("\n") for line in f]
    with open(uncompressed_trg) as f:
        corpus_trg = [line.rstrip("\n") for line in f]

    corpus_tsv = [f"{src_sent}\t{trg_sent}" for src_sent, trg_sent in zip(corpus_src, corpus_trg)]
    return corpus_tsv


def write_modified(modified: List[str], uncompressed_src: str, uncompressed_trg: str):
    """
    Split the modified TSV corpus, write back and compress
    """
    modified_src = "\n".join([line.split("\t")[0] for line in modified]) + "\n"
    modified_trg = "\n".join([line.split("\t")[1] for line in modified]) + "\n"

    with open(uncompressed_src, "w") as f:
        f.write(modified_src)
    with open(uncompressed_trg, "w") as f:
        f.writelines(modified_trg)

    # compress corpus back
    compress_file(uncompressed_src, keep_original=False)
    compress_file(uncompressed_trg, keep_original=False)


def run_import(
    type: str,
    dataset: str,
    output_prefix: str,
    src: str,
    trg: str,
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # these envs are standard across the pipeline

    if type == "corpus":
        # Parse a dataset identifier to extract importer, augmentation type and dataset name
        # Examples:
        # opus_wikimedia/v20230407
        # mtdata_EU-eac_forms-1-eng-lit
        # flores_aug-title_devtest
        # sacrebleu_aug-upper-strict_wmt19
        match = re.search(r"^(\w*)_(aug[a-z\-]*)?_?(.+)$", dataset)

        if not match:
            raise ValueError(
                f"Invalid dataset name: {dataset}. "
                f"Use the following format: <importer>_<name> or <importer>_<augmentation>_<name>."
            )

        importer = match.group(1)
        aug_modifer = match.group(2)
        name = match.group(3)

        no_aug_id = f"{importer}_{name}"

        print("Downloading parallel dataset")
        run_cmd(
            [os.path.join(current_dir, "download-corpus.sh"), no_aug_id, output_prefix],
            env={"SRC": src, "TRG": trg},
        )

        # TODO: convert everything to Chinese simplified for now
        # TODO: https://github.com/mozilla/firefox-translations-training/issues/896
        for lang in (src, trg):
            if lang == "zh":
                print("Converting the output file to Chinese Simplified")
                chinese_converter = ChineseConverter()
                stats = chinese_converter.convert_file(
                    Path(f"{output_prefix}.{lang}.zst"),
                    Path(f"{output_prefix}.converted.{lang}.zst"),
                    ChineseType.simplified,
                )
                shutil.move(f"{output_prefix}.converted.{lang}.zst", f"{output_prefix}.{lang}.zst")
                print(
                    f"Converted {stats.script_conversion.converted} lines from {stats.script_conversion.visited} to Chinese Simplified"
                )
                stats.save_json()

        if aug_modifer:
            print("Running augmentation")
            augment(output_prefix, aug_modifer, src=src, trg=trg)

    elif type == "mono":
        raise ValueError("Downloading mono data is not supported yet")
    else:
        raise ValueError(f"Invalid dataset type: {type}. Allowed values: mono, corpus")


def main() -> None:
    print(f"Running with arguments: {sys.argv}")
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--type", metavar="TYPE", type=str, help="Dataset type: mono or corpus")
    parser.add_argument(
        "--src",
        metavar="SRC",
        type=str,
        help="Source language",
    )
    parser.add_argument(
        "--trg",
        metavar="TRG",
        type=str,
        help="Target language",
    )
    parser.add_argument(
        "--dataset",
        metavar="DATASET",
        type=str,
        help="Full dataset identifier. For example, sacrebleu_aug-upper-strict_wmt19 ",
    )
    parser.add_argument(
        "--output_prefix",
        metavar="OUTPUT_PREFIX",
        type=str,
        help="Write output dataset to a path with this prefix",
    )

    args = parser.parse_args()
    print("Starting dataset import and augmentation.")
    run_import(args.type, args.dataset, args.output_prefix, args.src, args.trg)
    print("Finished dataset import and augmentation.")


if __name__ == "__main__":
    main()
