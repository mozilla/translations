import argparse
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Literal, Union

import humanize
import ruamel.yaml
from icu import Locale  # type: ignore

from pipeline.common.downloads import get_download_size, location_exists
from pipeline.data.cjk import CJK_LANGS
from pipeline.data.hplt import language_has_hplt_support
from utils.find_corpus import (
    fetch_mtdata,
    fetch_news_crawl,
    fetch_opus,
    fetch_sacrebleu,
    get_remote_file_size,
)

"""
Generate a training config for a language pair based on the latest production
training config, taskcluster/configs/config.prod.yml.
"""

root_dir = Path(__file__).parent.parent
prod_config_path = root_dir / "taskcluster/configs/config.prod.yml"

pretrained_student_models = {
    ("ru", "en"): "https://storage.googleapis.com/releng-translations-dev/models/ru-en/better-teacher/student"
}  # fmt: skip

skip_datasets = [
    # The NLLB dataset is based off of the CCMatrix dataset, and is mostly duplicated.
    "CCMatrix",
    # Skip Multi* datasets as they are generally multilingual versions of the original datasets.
    "MultiMaCoCu",
    "MultiHPLT",
    # In Russian, the WikiTitles data had its direction reversed. The `LinguaTools-WikiTitles`
    # version is fine.
    "WikiTitles",
    # This mtdata dataset fails in a task, and is a duplicate to OPUS.
    "swedish_work_environment",
    # Fails to load from mtdata.
    "lithuanian_legislation_seimas_lithuania",
    "ntrex",
    # Fails to load from OPUS.
    "SPC",
    # MTdata duplicates Flores that we pull directly
    "flores101_dev",
    "flores101_devtest",
    "flores200_dev",
    "flores200_devtest",
    # Skip OPUS WMT news test sets. They are used in our evaluation and shouldn't be used for training
    "WMT-News",
    # Contains blank lines.
    "wmt08",
]

# Do not include small datasets. This works around #508, and minimizes dataset tasks that
# won't bring a lot more data.
minimum_dataset_sentences = 200

# If a task name is too long, it will fail.
max_dataset_name_size = 80

flores_101_languages = {
    "af", "amh", "ar", "as", "ast", "az", "be", "bn", "bs", "bg", "ca", "ceb", "cs", "ckb", "cy",
    "da", "de", "el", "en", "et", "fa", "fi", "fr", "ful", "ga", "gl", "gu", "ha", "he", "hi",
    "hr", "hu", "hy", "ig", "id", "is", "it", "jv", "ja", "kam", "kn", "ka", "kk", "kea", "km",
    "ky", "ko", "lo", "lv", "ln", "lt", "lb", "lg", "luo", "ml", "mr", "mk", "mt", "mn", "mi",
    "ms", "my", "nl", "nb", "npi", "nso", "ny", "oc", "om", "or", "pa", "pl", "pt", "pus", "ro",
    "ru", "sk", "sl", "sna", "snd", "so", "es", "sr", "sv", "sw", "ta", "te", "tg", "tl", "th",
    "tr", "uk", "umb", "ur", "uz", "vi", "wo", "xh", "yo", "zh", "zh", "zu"
}  # fmt: skip

# mtdata points to raw downloads, and does some processing to normalize the data. This means
# that if we measure the download size, it may be inaccurate.
bad_mtdata_sizes = {
    # These are stored in a big archive with train/test/dev. Keep "train" estimates as they are
    # the largest, but ignore test/dev.
    "tedtalks_test",
    "tedtalks_dev",
}


def is_cjk(source: str, target: str) -> bool:
    return source in CJK_LANGS or target in CJK_LANGS


def get_git_revision_hash(remote_branch: str) -> str:
    """
    The git hash should be something that will always be around. Check the main branch for the
    most common ancestor to the local changes. The prod config locally could be different than
    remote, but it's better
    """
    return (
        subprocess.check_output(["git", "merge-base", remote_branch, "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_default_script(lang: str) -> str:
    locale = Locale(lang)
    # add default script
    locale = Locale.addLikelySubtags(locale)
    return locale.getScript()


def update_config(
    prod_config: Any, name: str, source: str, target: str, fast: bool
) -> dict[str, str]:
    experiment = prod_config["experiment"]

    # Update the prod config for this language pair.
    experiment["name"] = name
    experiment["src"] = source
    experiment["trg"] = target
    experiment["bicleaner"]["dataset-thresholds"] = {}
    # Use separate vocabs for languages with different scripts
    experiment["spm-vocab-split"] = get_default_script(source) != get_default_script(target)

    pretrained_model = pretrained_student_models.get((source, target))
    if pretrained_model:
        # Switch to the one stage teacher mode, as the higher quality backtranslations lead
        # to issues with early stopping when switching between stages.
        experiment["teacher-mode"] = "one-stage"
        prod_config["continuation"]["models"]["backwards"]["urls"] = [pretrained_model]
    else:
        prod_config["continuation"]["models"] = {}

    if is_cjk(source, target):
        experiment["spm-vocab-size"] = 64000
        experiment["opuscleaner-mode"] = "custom"

    datasets = prod_config["datasets"]

    # Clear out the base config.
    datasets["train"].clear()
    datasets["devtest"].clear()
    datasets["test"].clear()
    datasets["mono-src"].clear()
    datasets["mono-trg"].clear()

    # ruamel.yaml only supports inline comments. This dict will do string matching to apply
    # comments too the top of a section.
    comment_section = {}

    add_train_data(source, target, datasets, comment_section, fast)
    add_test_data(
        source,
        target,
        datasets["test"],
        datasets["devtest"],
        comment_section,
    )
    add_mono_data(
        source,
        "src",
        datasets,
        experiment,
        comment_section,
    )
    add_mono_data(
        target,
        "trg",
        datasets,
        experiment,
        comment_section,
    )

    return comment_section


def add_train_data(
    source: str,
    target: str,
    datasets: dict[str, list[str]],
    comment_section: dict[str, str],
    fast: bool,
):
    opus_datasets = fetch_opus(source, target)
    total_sentences = 0
    skipped_datasets: list[str] = []
    visited_corpora: set[str] = set()

    for dataset in opus_datasets:
        sentences = dataset.alignment_pairs or 0
        visited_corpora.add(normalize_corpus_name(dataset.corpus))

        # Some datasets are ignored or too small to be included.
        if dataset.corpus in skip_datasets:
            skipped_datasets.append(
                f"{dataset.corpus_key()} - ignored datasets ({sentences:,} sentences)"
            )
            continue
        if (dataset.alignment_pairs or 0) < minimum_dataset_sentences:
            skipped_datasets.append(
                f"{dataset.corpus_key()} - not enough data  ({sentences:,} sentences)"
            )
            continue
        if len(dataset.corpus) > max_dataset_name_size:
            skipped_datasets.append(f"{dataset.corpus_key()} - corpus name is too long for tasks")
            continue

        total_sentences += sentences
        corpus_key = dataset.corpus_key()
        datasets["train"].append(corpus_key)
        datasets["train"].yaml_add_eol_comment(  # type: ignore
            f"{sentences:,} sentences".rjust(70 - len(corpus_key), " "),
            len(datasets["train"]) - 1,
        )

    print("Fetching mtdata")
    entries = fetch_mtdata(source, target)

    for corpus_key, entry in entries.items():
        if entry.did.name in skip_datasets:
            continue
        modified_corpus_key = corpus_key

        # mtdata can have test and devtest data as well.
        if entry.did.name.endswith("test"):
            dataset = datasets["test"]
        elif entry.did.name.endswith("dev"):
            dataset_name = corpus_key[corpus_key.find("_") + 1 :]
            modified_corpus_key = f"mtdata_{aug_mix_modifier}_{dataset_name}"
            dataset = datasets["devtest"]
        else:
            dataset = datasets["train"]
            corpus_name = normalize_corpus_name(entry.did.name)
            group_corpus_name = normalize_corpus_name(entry.did.group + entry.did.name)
            if corpus_name in visited_corpora or group_corpus_name in visited_corpora:
                skipped_datasets.append(f"{corpus_key} - duplicate with opus")
                continue

            if entry.did.name in skip_datasets:
                skipped_datasets.append(f"{entry.did.name} - ignored datasets")
                continue
            if len(entry.did.name) > max_dataset_name_size:
                skipped_datasets.append(f"{entry.did.name} - corpus name is too long for tasks")
                continue

        if fast:
            # Just add the dataset when in fast mode.
            dataset.append(modified_corpus_key)
        else:
            byte_size = None
            display_size = None
            if isinstance(entry.url, tuple):
                size_a = get_remote_file_size(entry.url[0])[0]
                size_b = get_remote_file_size(entry.url[1])[0]
                if size_a and size_b:
                    byte_size = size_a + size_b
                    display_size = humanize.naturalsize(byte_size)

            else:
                byte_size, display_size = get_remote_file_size(entry.url)

            if byte_size is None:
                # There was a network error, skip the dataset.
                skipped_datasets.append(f"{corpus_key} - Error fetching ({entry.url})")
            else:
                # Don't add the sentences to the total_sentences, as mtdata is less reliable
                # compared to opus.
                sentences = estimate_sentence_size(byte_size)
                if byte_size:
                    dataset.append(modified_corpus_key)
                    dataset.yaml_add_eol_comment(  # type: ignore
                        f"~{sentences:,} sentences ".rjust(70 - len(corpus_key), " ")
                        + f"({display_size})",
                        len(datasets["train"]) - 1,
                    )
                else:
                    skipped_datasets.append(
                        f"{corpus_key} - No Content-Length reported ({entry.url})"
                    )

    comments = [
        "The training data contains:",
        f"  {total_sentences:,} sentences",
    ]
    if skipped_datasets:
        comments.append("")
        comments.append("Skipped datasets:")
        for d in skipped_datasets:
            comments.append(f" - {d}")

    train_comment = "\n".join(comments)

    comment_section["  train:"] = train_comment


def normalize_corpus_name(corpus_name: str):
    """Normalize the corpus name so that it's easy to deduplicate between opus and mtdata."""

    # Remove the language tags at the end.
    # mtdata_ELRC-vnk.fi-1-eng-fin
    #                     ^^^^^^^^
    corpus_name = re.sub(r"-\w{3}-\w{3}$", "", corpus_name)

    corpus_name = corpus_name.lower()

    # Remove numbers anything that is not a letter. This is a little aggressive, but should help
    # deduplicate more datasets. For example:
    #   opus: 725-Hallituskausi_2011_2
    #   mtdata: hallituskausi_2011_2015-1-eng-fin
    corpus_name = re.sub(r"[^a-z]", "", corpus_name.lower())

    # Datasets could be split by train/test/dev. Remove the "train" word so that it will match
    # between Opus and mtdata.
    #   opus: NeuLab-TedTalks/v1
    #   mtdata: Neulab-tedtalks_train-1-eng-fin
    #   mtdata: Neulab-tedtalks_test-1-eng-fin
    #   mtdata: Neulab-tedtalks_dev-1-eng-fin
    corpus_name = re.sub(r"train$", "", corpus_name)

    return corpus_name


def add_test_data(
    source: str,
    target: str,
    test_datasets: list[str],
    devtest_datasets: list[str],
    comment_section: dict[str, str],
):
    skipped_datasets = []
    print("Fetching flores")
    if source in flores_101_languages and target in flores_101_languages:
        test_datasets.append("flores_devtest")

        # Add augmented datasets to check performance for the specific cases
        devtest_datasets.append("flores_aug-mix_dev")
        test_datasets.append("flores_aug-mix_devtest")
        test_datasets.append("flores_aug-noise_devtest")
        test_datasets.append("flores_aug-inline-noise_devtest")
        test_datasets.append("flores_aug-punct_devtest")
        if not is_cjk(source, target):
            test_datasets.append("flores_aug-title_devtest")
            test_datasets.append("flores_aug-upper_devtest")
            test_datasets.append("flores_aug-typos_devtest")

    is_test = True  # Flip between devtest and test.
    print("Fetching sacrebleu")
    for d in fetch_sacrebleu(source, target):
        # Work around: PLW2901 `for` loop variable `dataset_name` overwritten by assignment target
        dataset_name = d
        if dataset_name in skip_datasets:
            # This could be a dataset with a variant design.
            skipped_datasets.append(f"{dataset_name} - variant dataset")
        elif len(dataset_name) > max_dataset_name_size:
            skipped_datasets.append(f"{dataset_name} - corpus name is too long for tasks")
        else:
            dataset_name = dataset_name.replace("sacrebleu_", "")
            if is_test:
                test_datasets.append(f"sacrebleu_{dataset_name}")
            else:
                devtest_datasets.append(f"sacrebleu_aug-mix_{dataset_name}")
            is_test = not is_test

    if skipped_datasets:
        test_comment = "\n".join(
            [
                "Skipped test/devtest datasets:",
                *[f" - {d}" for d in skipped_datasets],
            ]
        )

        comment_section["  devtest:"] = test_comment


def estimate_sentence_size(bytes: int) -> int:
    """Estimate the sentences based on the compressed byte size"""
    # One dataset measured 113 bytes per sentence, use that as a rough estimate.
    bytes_per_sentence = 113
    return bytes // bytes_per_sentence


def add_mono_data(
    lang: str,
    direction: Union[Literal["src"], Literal["trg"]],
    datasets: dict[str, list[str]],
    experiment: Any,
    comment_section: dict[str, str],
):
    mono_datasets = datasets[f"mono-{direction}"]
    max_per_dataset: int = experiment[f"mono-max-sentences-{direction}"]["per-dataset"]

    def add_comment(dataset_name: str, comment: str):
        """Add a right justified comment to a dataset."""
        mono_datasets.yaml_add_eol_comment(  # type: ignore
            comment.rjust(50 - len(dataset_name), " "),
            len(mono_datasets) - 1,
        )

    extra_comments: list[str] = []
    skipped_datasets = []

    print("Fetching newscrawl for", lang)
    sentence_count = 0
    for dataset in fetch_news_crawl(lang):
        mono_datasets.append(dataset.name)
        if dataset.size:
            sentences = estimate_sentence_size(dataset.size)
            sentence_count += sentences
            add_comment(dataset.name, f"~{sentences:,} sentences")

    print("Fetching HPLT mono for", lang)
    if language_has_hplt_support(lang):
        dataset_name = "hplt_mono/v2.0"
        mono_datasets.append(dataset_name)
        add_comment(dataset_name, f"Up to {max_per_dataset:,} sentences")
        extra_comments.append(f"  Up to {max_per_dataset:,} sentences from HPLT")

    print("Fetching NLLB mono for", lang)
    opus_nllb_url = f"https://object.pouta.csc.fi/OPUS-NLLB/v1/mono/{lang}.txt.gz"
    if location_exists(opus_nllb_url):
        dataset_name = "opus_NLLB/v1"
        lines_num = estimate_sentence_size(get_download_size(opus_nllb_url))
        mono_datasets.append(dataset_name)
        sentence_count += lines_num
        add_comment(dataset_name, f"~{lines_num:,} sentences")

    skipped_datasets_final = []
    if skipped_datasets:
        skipped_datasets_final.append("")
        skipped_datasets_final.append("Skipped datasets:")
        for d in skipped_datasets:
            skipped_datasets_final.append(f" - {d}")

    comment = "\n".join(
        [
            "The monolingual data contains:",
            f"  ~{sentence_count:,} sentences",
            # Append any additional information.
            *extra_comments,
            *skipped_datasets_final,
        ]
    )

    comment_section[f"  mono-{direction}:"] = comment


def strip_comments(yaml_text: str) -> str:
    """
    ruamel.yaml preserves key ordering and comments. This function strips out the comments

    """
    result = ""
    for l in yaml_text.splitlines():
        # Work around: PLW2901 `for` loop variable `line` overwritten by assignment target
        line = l
        if line.strip().startswith("#"):
            continue

        # Remove any comments at the end.
        line = re.sub(r"#[\s\w\-.]*$", "", line)

        # Don't add any empty lines.
        if line.strip():
            result += line.rstrip() + "\n"

    return result


def apply_comments_to_yaml_string(yaml, prod_config, comment_section, remote_branch: str) -> str:
    """
    ruamel.yaml only supports inline comments, so do direct string manipulation to apply
    all the comments needed.
    """
    # Dump out the yaml to a string so that it can be manipulated.
    output_stream = StringIO()
    yaml.dump(prod_config, output_stream)
    yaml_string: str = output_stream.getvalue()
    yaml_string = apply_comment_section(comment_section, yaml_string)

    script_args = " ".join(sys.argv[1:])
    return "\n".join(
        [
            "# The initial configuration was generated using:",
            f"# task config-generator -- {script_args}",
            "#",
            "# The documentation for this config can be found here:",
            f"# https://github.com/mozilla/translations/blob/{get_git_revision_hash(remote_branch)}/taskcluster/configs/config.prod.yml",
            yaml_string,
        ]
    )


def apply_comment_section(comment_section: dict[str, str], yaml_string: str) -> str:
    for key, raw_comment in comment_section.items():
        # Find the indent amount for the key.
        match = re.search(r"^(?P<indent>\s*)", key)
        if not match:
            raise Exception("Could not find regex match")
        indent = match.group("indent")

        # Indent the lines, and add the # comment.
        comment = "\n".join([f"{indent}# {line}" for line in raw_comment.splitlines()])

        yaml_string = yaml_string.replace(f"\n{key}", f"\n\n{comment}\n{key}")
    return yaml_string


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("source", metavar="SOURCE", type=str, help="The source language tag")
    parser.add_argument("target", metavar="TARGET", type=str, help="The target language tag")
    parser.add_argument(
        "--name",
        metavar="name",
        type=str,
        required=True,
        help="The name of the config, which gets constructed like so: configs/autogenerated/{source}-{target}-{name}.yml",
    )
    parser.add_argument(
        "--remote_branch",
        metavar="REF",
        type=str,
        default="origin/main",
        help="The remote branch that contains the config.prod.yml. Typically origin/main, or origin/release",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow network requests like looking up dataset size",
    )

    args = parser.parse_args()

    source: str = args.source
    target: str = args.target
    name: str = args.name
    remote_branch: str = args.remote_branch
    fast: bool = args.fast

    # Validate the inputs.
    langtag_re = r"[a-z]{2,3}"
    if not re.fullmatch(langtag_re, source):
        print("The source language should be a 2 or 3 letter lang tag.")
    if not re.fullmatch(langtag_re, target):
        print("The target language should be a 2 or 3 letter lang tag.")
    if not re.fullmatch(r"[\w\d-]+", name):
        print(
            "The name of the training config should only contain alphanumeric, underscores, and dashes.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ruamel.yaml preserves comments and ordering unlink PyYAML
    yaml = ruamel.yaml.YAML()

    # Load the prod yaml.
    with prod_config_path.open() as f:
        yaml_string = f.read()
    yaml_string = strip_comments(yaml_string)
    prod_config = yaml.load(StringIO(yaml_string))

    comment_section = update_config(prod_config, name, source, target, fast)
    final_config = apply_comments_to_yaml_string(yaml, prod_config, comment_section, remote_branch)
    config_dir = root_dir / "configs" / name
    config_dir.mkdir(exist_ok=True)
    final_config_path = config_dir / f"{source}-{target}.yml"

    print("Writing config to:", str(final_config_path))
    final_config_path.write_text(final_config)


if __name__ == "__main__":
    main()
