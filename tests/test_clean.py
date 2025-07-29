import os
from pathlib import Path

import pytest
import zstandard as zstd

from fixtures import DataDir, get_mocked_downloads, FIXTURES_PATH
from pipeline.data import parallel_importer

SRC = "ru"
TRG = "en"
CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def add_fake_alignments(corpus):
    corpus_and_aln = []
    for line in corpus:
        parts = line.split("\t")
        src_sent, trg_sent = parts[0], parts[1]
        min_len = min(len(src_sent.split()), len(trg_sent.split()))
        aln = " ".join([f"{idx}-{idx}" for idx in range(min_len)])
        corpus_and_aln.append(f"{line}\t{aln}")

    return corpus_and_aln


# it's very slow to download and run BERT on 2000 lines
parallel_importer.add_alignments = add_fake_alignments


def config(trg_lang, data_dir):
    if trg_lang == "zh":
        config_path = os.path.abspath(os.path.join(FIXTURES_PATH, "config.pytest.enzh.yml"))
    elif trg_lang == "fr":
        config_path = os.path.abspath(os.path.join(FIXTURES_PATH, "config.pytest.enfr.yml"))
    else:
        # copy the test config and swap language direction
        config_path = os.path.abspath(os.path.join(FIXTURES_PATH, "config.pytest.yml"))

    new_config_path = data_dir.join(f"config.{trg_lang}.yml")

    with open(config_path) as f:
        new_config = f.read()

    if trg_lang == "en":
        new_config = new_config.replace(
            """  src: en
trg: ru""",
            """  src: ru
trg: en""",
        )

    # disable monocleaner
    new_config = new_config.replace("default-threshold: 0.5", "default-threshold: 0.0").replace(
        "default-threshold: 0.9", "default-threshold: 0.0"
    )

    with open(new_config_path, "w") as f:
        f.write(new_config)

    return new_config_path


def read_lines(path):
    with zstd.open(path, "rt") as f:
        return f.readlines()


@pytest.fixture(scope="function")
def data_dir():
    return DataDir("test_clean")


@pytest.mark.parametrize(
    "importer,src_lang,trg_lang,dataset",
    [
        # tests langauge-specific config
        ("opus", "en", "zh", "NeuLab-TedTalks_v1"),
        # tests dataset-specific config
        ("opus", "en", "ru", "ELRC-3075-wikipedia_health_v1"),
        # tests default config
        ("opus", "en", "ru", "ELRC_2922_v1"),
        # tests Latin to Latin
        ("opus", "en", "fr", "ELRA-W0308_v1"),
    ],
)
def test_clean_parallel(importer, src_lang, trg_lang, dataset, data_dir):
    # Download a small but real dataset using mock downloads
    data_dir.run_task(
        f"dataset-{importer}-{dataset}-{src_lang}-{trg_lang}",
        env={
            "WGET": os.path.join(CURRENT_FOLDER, "fixtures/wget"),
            "MOCKED_DOWNLOADS": get_mocked_downloads(),
        },
        config=config(trg_lang, data_dir),
    )

    artifacts_prefix = data_dir.join(f"artifacts/{dataset}")
    output_src = f"{artifacts_prefix}.{src_lang}.zst"
    output_trg = f"{artifacts_prefix}.{trg_lang}.zst"
    assert os.path.exists(output_src)
    assert os.path.exists(output_trg)
    src_lines = read_lines(output_src)
    trg_lines = read_lines(output_trg)
    assert len(src_lines) == len(trg_lines)
    # Move output artefacts to input fetches
    fetches_prefix = data_dir.join(f"{dataset}")
    Path(output_src).replace(f"{fetches_prefix}.{src_lang}.zst")
    Path(output_trg).replace(f"{fetches_prefix}.{trg_lang}.zst")

    # Run cleaning
    data_dir.run_task(
        f"corpus-clean-parallel-{importer}-{dataset}-{src_lang}-{trg_lang}",
        config=config(trg_lang, data_dir),
    )

    assert os.path.exists(output_src)
    assert os.path.exists(output_trg)
    src_filtered_lines = read_lines(output_src)
    trg_filtered_lines = read_lines(output_trg)
    assert len(src_filtered_lines) == len(trg_filtered_lines)
    # something was filtered but not everything
    assert len(src_filtered_lines) > 0
    assert len(src_filtered_lines) < len(src_lines)


@pytest.mark.parametrize(
    "target_language",
    ["ru", "zh"],
)
def test_clean_mono(target_language, data_dir):
    importer = "news-crawl"
    dataset = "news_2021"
    data_dir.run_task(
        f"dataset-{importer}-{dataset}-{target_language}",
        env={
            "WGET": os.path.join(CURRENT_FOLDER, "fixtures/wget"),
            "MOCKED_DOWNLOADS": get_mocked_downloads(),
        },
        config=config(target_language, data_dir),
    )

    artifacts_prefix = data_dir.join(f"artifacts/{dataset}")
    output = f"{artifacts_prefix}.{target_language}.zst"
    assert os.path.exists(output)
    original_lines = read_lines(output)
    # Move output artifacts to input fetches
    fetches_prefix = data_dir.join(f"{dataset}")
    Path(output).replace(f"{fetches_prefix}.{target_language}.zst")

    # Run cleaning
    data_dir.run_task(
        f"corpus-clean-mono-{importer}-{target_language}-{dataset}-mono-trg",
        config=config(target_language, data_dir),
    )

    assert os.path.exists(output)
    filtered_lines = read_lines(output)
    # something might've been filtered
    assert len(filtered_lines) > 0
    assert len(filtered_lines) <= len(original_lines)
