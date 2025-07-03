import os
import shutil
from typing import Any, Optional

import pytest
import sentencepiece as spm
from fixtures import DataDir, en_sample, ru_sample, zh_sample, FIXTURES_PATH

pytestmark = [pytest.mark.docker_amd64]

current_folder = os.path.dirname(os.path.abspath(__file__))
fixtures_path = os.path.join(current_folder, "fixtures")
root_path = os.path.abspath(os.path.join(current_folder, ".."))
bin_dir = os.environ["BIN"] if os.getenv("BIN") else os.path.join(root_path, "bin")
marian_dir = (
    os.environ["MARIAN"]
    if os.getenv("MARIAN")
    else os.path.join(root_path, "3rd_party", "marian-dev", "build")
)


def validate_alignments(corpus_path: str, vocab_src_path: str, vocab_trg_path: str) -> None:
    # This module is not well-typed, set it to Any.
    SentencePieceProcessor: Any = spm.SentencePieceProcessor
    sp_src = SentencePieceProcessor(model_file=vocab_src_path)
    sp_trg = SentencePieceProcessor(model_file=vocab_trg_path)

    print("Validating alignments:", corpus_path)
    with open(corpus_path) as f:
        for line in f:
            fields = line.strip().split("\t")
            assert len(fields) == 3, "The alignments should be retained."
            src = sp_src.encode_as_pieces(fields[0])
            trg = sp_trg.encode_as_pieces(fields[1])
            alignment = [[int(num) for num in pair.split("-")] for pair in fields[2].split()]

            for idx_src, idx_trg in alignment:
                try:
                    assert src[idx_src] is not None
                    assert trg[idx_trg] is not None
                except:
                    print("src: ", src)
                    print("trg: ", trg)
                    print("alignment:", alignment)
                    raise


@pytest.fixture()
def data_dir() -> DataDir:
    return DataDir("test_training")


@pytest.fixture(params=["ru", "zh"])
def trg_lang(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def config(trg_lang: str) -> Optional[str]:
    zh_config_path = os.path.abspath(os.path.join(FIXTURES_PATH, "config.pytest.enzh.yml"))
    return zh_config_path if trg_lang == "zh" else None


@pytest.fixture()
def vocab(data_dir: DataDir, trg_lang: str) -> tuple[str, str]:
    output_path_src = data_dir.join("vocab.en.spm")
    output_path_trg = data_dir.join(f"vocab.{trg_lang}.spm")
    vocab_path = "tests/data/vocab.spm" if trg_lang == "ru" else "tests/data/vocab.zhen.spm"
    shutil.copyfile(vocab_path, output_path_src)
    shutil.copyfile(vocab_path, output_path_trg)
    print(f"Using vocab {vocab_path}")

    return output_path_src, output_path_trg


@pytest.fixture()
def corpus_files(data_dir: DataDir, trg_lang: str):
    sample = zh_sample if trg_lang == "zh" else ru_sample
    data_dir.create_zst("corpus.en.zst", en_sample)
    data_dir.create_zst(f"corpus.{trg_lang}.zst", sample)
    data_dir.create_zst("mono.en.zst", en_sample)
    data_dir.create_zst(f"mono.{trg_lang}.zst", sample)
    data_dir.create_zst("devset.en.zst", en_sample)
    data_dir.create_zst(f"devset.{trg_lang}.zst", sample)


@pytest.fixture()
def alignments(
    data_dir: DataDir,
    vocab: tuple[str, str],
    corpus_files: None,
    trg_lang: str,
    config: Optional[str],
) -> None:
    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "MARIAN": marian_dir,
        "SRC": "en",
        "TRG": trg_lang,
        "USE_CPU": "true",
    }
    for task, corpus in [("parallel", "corpus"), ("backtranslations", "mono")]:
        data_dir.run_task(f"corpus-align-{task}-en-{trg_lang}", env=env, config=config)

        shutil.copyfile(
            data_dir.join("artifacts", f"{corpus}.aln.zst"),
            data_dir.join(f"{corpus}.aln.zst"),
        )
        for lang in ["en", trg_lang]:
            shutil.copyfile(
                data_dir.join("artifacts", f"{corpus}.tok-icu.{lang}.zst"),
                data_dir.join(f"{corpus}.tok-icu.{lang}.zst"),
            )
        if task == "parallel":
            shutil.copyfile(
                data_dir.join("artifacts", "corpus.priors"),
                data_dir.join("corpus.priors"),
            )
    # recreate corpus
    data_dir.create_zst("corpus.en.zst", en_sample)
    sample = zh_sample if trg_lang == "zh" else ru_sample
    data_dir.create_zst(f"corpus.{trg_lang}.zst", sample)

    # The artifacts should be removed so that it's clear what the train task generated.
    shutil.rmtree(data_dir.join("artifacts"))


def test_train_student_mocked(
    alignments: None,
    data_dir: DataDir,
    trg_lang: str,
    vocab: tuple[str, str],
    config: Optional[str],
):
    """
    Run training with mocked marian to check OpusTrainer output
    """

    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "MARIAN": fixtures_path,
        "SRC": "en",
        "TRG": trg_lang,
        "USE_CPU": "true",
    }

    data_dir.run_task(f"distillation-student-model-train-en-{trg_lang}", env=env, config=config)
    data_dir.print_tree()

    assert os.path.isfile(data_dir.join("artifacts", "final.model.npz.best-chrf.npz"))
    assert os.path.isfile(data_dir.join("artifacts", "model.npz.best-chrf.npz.decoder.yml"))

    validate_alignments(data_dir.join("marian.input.txt"), vocab[0], vocab[1])


def test_train_student(alignments: None, data_dir: DataDir, trg_lang: str, config: Optional[str]):
    """
    Run real training with Marian as an integration test
    """

    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "MARIAN": marian_dir,
        "SRC": "en",
        "TRG": trg_lang,
        "USE_CPU": "true",
        "WORKSPACE": "2000",
    }
    marian_args = [
        "--disp-freq", "1",
        "--save-freq", "2",
        "--valid-freq", "2",
        "--after-batches", "2",
        "--dim-vocabs", "1000", "1000",
        "--mini-batch", "10",
        "--maxi-batch", "10",
        "--mini-batch-fit", "false",
        "--log-level", "trace",
    ]  # fmt:skip

    data_dir.run_task(
        f"distillation-student-model-train-en-{trg_lang}",
        env=env,
        extra_args=marian_args,
        config=config,
    )
    data_dir.print_tree()

    assert os.path.isfile(data_dir.join("artifacts", "final.model.npz.best-chrf.npz"))
    assert os.path.isfile(data_dir.join("artifacts", "model.npz.best-chrf.npz.decoder.yml"))


def test_train_teacher(
    alignments: None, data_dir: DataDir, trg_lang: str, config: Optional[str]
) -> None:
    """
    Run real training with Marian as an integration test
    """

    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "MARIAN": marian_dir,
        "SRC": "en",
        "TRG": trg_lang,
        "USE_CPU": "true",
        "WORKSPACE": "2000",
    }
    marian_args = [
        "--disp-freq", "1",
        "--save-freq", "2",
        "--valid-freq", "2",
        "--after-batches", "2",
        "--dim-vocabs", "1000", "1000",
        "--mini-batch", "10",
        "--maxi-batch", "10",
        "--mini-batch-fit", "false",
        "--log-level", "trace",
        # simplify architecture to run faster
        "--task", "transformer-base",
        "--enc-depth", "1",
        "--dec-depth", "1",
        "--dim-emb", "32",
        "--transformer-dim-ffn", "128",
        "--beam-size", "1"

    ]  # fmt:skip

    data_dir.run_task(
        f"train-teacher-model-en-{trg_lang}-1", env=env, extra_args=marian_args, config=config
    )

    assert os.path.isfile(data_dir.join("artifacts", "final.model.npz.best-chrf.npz"))
    assert os.path.isfile(data_dir.join("artifacts", "model.npz.best-chrf.npz.decoder.yml"))


def test_train_backwards(
    corpus_files: None,
    vocab: tuple[str, str],
    data_dir: DataDir,
    trg_lang: str,
    config: Optional[str],
):
    """
    Run real training with Marian as an integration test
    """

    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "MARIAN": marian_dir,
        "SRC": "en",
        "TRG": trg_lang,
        "USE_CPU": "true",
        "WORKSPACE": "2000",
    }
    marian_args = [
        "--disp-freq", "1",
        "--save-freq", "2",
        "--valid-freq", "2",
        "--after-batches", "2",
        "--dim-vocabs", "1000", "1000",
        "--mini-batch", "10",
        "--maxi-batch", "10",
        "--mini-batch-fit", "false",
        "--log-level", "trace",
        "--enc-depth", "1",
        "--dec-depth", "1",
        "--dim-emb", "32",
        "--dim-rnn", "16",
        "--beam-size", "1"
    ]  # fmt:skip

    data_dir.run_task(
        f"backtranslations-train-backwards-model-en-{trg_lang}",
        env=env,
        extra_args=marian_args,
        config=config,
    )

    assert os.path.isfile(data_dir.join("artifacts", "final.model.npz.best-chrf.npz"))
    assert os.path.isfile(data_dir.join("artifacts", "model.npz.best-chrf.npz.decoder.yml"))


def test_train_backwards_mocked(
    data_dir: DataDir,
    vocab: tuple[str, str],
    corpus_files: None,
    trg_lang: str,
    config: Optional[str],
):
    """
    Run training with mocked Marian to validate the backwards training configuration.
    """

    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "MARIAN": fixtures_path,
        "SRC": "en",
        "TRG": trg_lang,
        "USE_CPU": "true",
    }

    data_dir.run_task(
        f"backtranslations-train-backwards-model-en-{trg_lang}", env=env, config=config
    )
    data_dir.print_tree()

    assert os.path.isfile(data_dir.join("artifacts", "final.model.npz.best-chrf.npz"))
    assert os.path.isfile(data_dir.join("artifacts", "model.npz.best-chrf.npz.decoder.yml"))


def test_train_teacher_mocked(
    alignments: None, data_dir: DataDir, trg_lang: str, config: Optional[str]
):
    """
    Run training with mocked Marian to validate the teacher training configuration.
    """

    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "MARIAN": fixtures_path,
        "SRC": "en",
        "TRG": trg_lang,
        "USE_CPU": "true",
    }

    # Fake the datasets that are required.
    shutil.copy(data_dir.join("corpus.aln.zst"), data_dir.join("mono.aln.zst"))
    shutil.copy(data_dir.join("corpus.en.zst"), data_dir.join("mono.en.zst"))
    shutil.copy(data_dir.join(f"corpus.{trg_lang}.zst"), data_dir.join(f"mono.{trg_lang}.zst"))

    data_dir.run_task(f"train-teacher-model-en-{trg_lang}-1", env=env, config=config)
    data_dir.print_tree()

    assert os.path.isfile(data_dir.join("artifacts", "final.model.npz.best-chrf.npz"))
