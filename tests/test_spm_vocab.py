import os
import re
import subprocess
import sys

import pytest
from fixtures import DataDir, en_sample, ru_sample

current_folder = os.path.dirname(os.path.abspath(__file__))
fixtures_path = os.path.join(current_folder, "fixtures")
norm_rule_path = "pipeline/train/normalization/nmt_nfkc.tsv"


def run_spm_test(arguments: list[str]) -> list[str]:
    """
    Run the training script and return the spm_train arguments.
    """

    # See data/test_data/test_spm_vocab for the artifacts after a failure.
    test_data_dir = DataDir("test_spm_vocab")

    env = {
        **os.environ,
        "MARIAN": fixtures_path,
        # This allows the spm_train fixture to know where to output the vocab.
        "SPM_VOCAB_DATA_DIRECTORY": test_data_dir.path,
    }
    command = [
        "pipeline/train/spm-vocab.sh",
        test_data_dir.create_zst("corpus.en.zst", en_sample),
        test_data_dir.create_zst("corpus.ru.zst", ru_sample),
        test_data_dir.join("vocab.en.spm"),
        test_data_dir.join("vocab.ru.spm"),
        *arguments,
    ]

    result = subprocess.run(command, env=env, stderr=subprocess.PIPE, check=False)

    # On failure surface the stderr as an Exception.
    if not result.returncode == 0:
        print(result.stderr, file=sys.stderr)
        raise Exception(result.stderr)

    for lang in ("en", "ru"):
        vocab_path = test_data_dir.join(f"vocab.{lang}.spm")
        if not os.path.exists(vocab_path):
            raise Exception(f"The vocab file for {lang} was not processed.")

        with open(vocab_path, "r", encoding="utf-8") as file:
            return file.read()


def test_no_vocab_size():
    spm_train_arguments = run_spm_test(["1000", "auto", "false", norm_rule_path])
    assert "--vocab_size=32000" in spm_train_arguments, "The vocab size is set to the default."
    assert (
        "vocab.src" not in spm_train_arguments and "vocab.trg" not in spm_train_arguments
    ), "Do not use split vocabs"
    assert (
        "--input_sentence_size=1000" in spm_train_arguments
    ), "The input sentence size is respected."
    assert re.search(
        r"--num_threads\s+\d+", spm_train_arguments
    ), "The number of threads is automatically set."
    assert re.search(
        r"--normalization_rule_tsv=.*nmt_nfkc.tsv", spm_train_arguments
    ), "The normalization rule is set."


def test_none_vocab_size():
    """Taskcluster can provide the argument "None" rather than an empty variable."""
    spm_train_arguments = run_spm_test(["1000", "auto", "false", norm_rule_path, "None"])
    assert "--vocab_size=32000" in spm_train_arguments, "The vocab size is set to the default."
    assert (
        "vocab.src" not in spm_train_arguments and "vocab.trg" not in spm_train_arguments
    ), "Do not use split vocabs"
    assert (
        "--input_sentence_size=1000" in spm_train_arguments
    ), "The input sentence size is respected."
    assert re.search(
        r"--num_threads\s+\d+", spm_train_arguments
    ), "The number of threads is automatically set."
    assert re.search(
        r"--normalization_rule_tsv=.*nmt_nfkc.tsv", spm_train_arguments
    ), "The normalization rule is set."


def test_vocab_fully_specified():
    """Fully specify all the values."""
    spm_train_arguments = run_spm_test(["3333", "4", "false", norm_rule_path, "1024"])
    assert "--vocab_size=1024" in spm_train_arguments, "The vocab size is specified."
    assert (
        "vocab.src" not in spm_train_arguments and "vocab.trg" not in spm_train_arguments
    ), "Do not use split vocabs"
    assert (
        "--input_sentence_size=3333" in spm_train_arguments
    ), "The input sentence size is respected."
    assert "--num_threads\n4" in spm_train_arguments, "The number of threads is manually set."
    assert re.search(
        r"--normalization_rule_tsv=.*nmt_nfkc.tsv", spm_train_arguments
    ), "The normalization rule is set."


def test_non_multiples_eight():
    """Non-multiples of 8 fail for the vocab size."""
    with pytest.raises(Exception) as exception_info:
        run_spm_test(["3333", "4", "false", norm_rule_path, "13"])

    assert "vocab_size must be a multiple of 8" in str(exception_info.value)


def test_split_vocab():
    """Fully specify all the values."""
    spm_train_arguments = run_spm_test(["3333", "4", "true", norm_rule_path, "1024"])
    assert "--vocab_size=1024" in spm_train_arguments, "The vocab size is specified."
    assert (
        "vocab.src" in spm_train_arguments or "vocab.trg" in spm_train_arguments
    ), "Use split vocabs"
    assert (
        "--input_sentence_size=3333" in spm_train_arguments
    ), "The input sentence size is respected."
    assert "--num_threads\n4" in spm_train_arguments, "The number of threads is manually set."
    assert re.search(
        r"--normalization_rule_tsv=.*nmt_nfkc.tsv", spm_train_arguments
    ), "The normalization rule is set."


def test_no_norm_rule():
    """Fully specify all the values."""
    spm_train_arguments = run_spm_test(["3333", "4", "false", "None", "1024"])
    assert "--vocab_size=1024" in spm_train_arguments, "The vocab size is specified."
    assert (
        "vocab.src" not in spm_train_arguments and "vocab.trg" not in spm_train_arguments
    ), "Do not use split vocabs"
    assert (
        "--input_sentence_size=3333" in spm_train_arguments
    ), "The input sentence size is respected."
    assert "--num_threads\n4" in spm_train_arguments, "The number of threads is manually set."
    assert re.search(
        r"--normalization_rule_tsv=.*nmt_nfkc.tsv", spm_train_arguments
    ), "The normalization rule is set."
