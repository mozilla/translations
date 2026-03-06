import json
from pathlib import Path
import shutil

import pytest
from fixtures import DataDir, en_sample
from pipeline.common.marian import marian_args_to_dict

fixtures_path = Path(__file__).parent / "fixtures"


@pytest.fixture
def data_dir():
    data_dir = DataDir("test_translate")
    shutil.copyfile("tests/data/vocab.spm", data_dir.join("vocab.en.spm"))
    shutil.copyfile("tests/data/vocab.spm", data_dir.join("vocab.ru.spm"))
    return data_dir


def sanitize_marian_args(args_list: list[str]):
    """
    Marian args can have details that reflect the host machine or are unique per run.
    Sanitize those here.
    """
    base_dir = str((Path(__file__).parent / "..").resolve())
    args_dict = marian_args_to_dict(args_list)
    for key, value in args_dict.items():
        if isinstance(value, list):
            for index, value_inner in enumerate(value):
                if isinstance(value_inner, str):
                    if value_inner.startswith("/tmp"):
                        value[index] = "<tmp>/" + Path(value_inner).name
                    if value_inner.startswith(base_dir):
                        value[index] = value_inner.replace(base_dir, "<src>")
        elif isinstance(value, str):
            if value.startswith("/tmp"):
                args_dict[key] = "<tmp>/" + Path(value).name
            if value.startswith(base_dir):
                args_dict[key] = value.replace(base_dir, "<src>")

    return args_dict


def test_translate_corpus(data_dir: DataDir):
    data_dir.create_zst("file.1.zst", en_sample)
    data_dir.create_file("fake-model.npz", "")
    data_dir.run_task(
        "distillation-parallel-src-translate-en-ru-1/10",
        env={
            "MARIAN": str(fixtures_path),
            "TEST_ARTIFACTS": data_dir.path,
            "USE_CPU": "1",
        },
    )
    data_dir.print_tree()

    output = data_dir.read_text("artifacts/file.1.nbest.zst")
    for pseudo_translated in en_sample.upper().split("\n"):
        assert pseudo_translated in output

    args = json.loads(data_dir.read_text("marian-decoder.args.txt"))
    assert sanitize_marian_args(args) == {
        "config": "<src>/pipeline/translate/decoder.yml",
        "vocabs": [
            "<src>/data/tests_data/test_translate/vocab.en.spm",
            "<src>/data/tests_data/test_translate/vocab.ru.spm",
        ],
        "input": "<tmp>/file.1",
        "output": "<tmp>/file.1.nbest",
        "n-best": True,
        "log": "<tmp>/file.1.log",
        "devices": ["0", "1", "2", "3"],
        "workspace": "12000",
        "mini-batch-words": "4000",
        "precision": "float16",
        "models": "<src>/data/tests_data/test_translate/fake-model.npz",
    }


def test_translate_corpus_empty(data_dir: DataDir):
    """
    Test the case of an empty file.
    """
    data_dir.create_zst("file.1.zst", "")
    data_dir.create_file("fake-model.npz", "")
    data_dir.run_task(
        "distillation-parallel-src-translate-en-ru-1/10",
        env={
            "MARIAN": str(fixtures_path),
            "TEST_ARTIFACTS": data_dir.path,
            "USE_CPU": "1",
        },
    )

    data_dir.print_tree()

    assert data_dir.read_text("artifacts/file.1.nbest.zst") == "", "The text is empty"


@pytest.mark.parametrize(
    "params",
    [
        (
            # task
            "distillation-mono-src-translate",
            # marian_args
            {
                "config": "<src>/pipeline/translate/decoder.yml",
                "vocabs": [
                    "<src>/data/tests_data/test_translate/vocab.en.spm",
                    "<src>/data/tests_data/test_translate/vocab.ru.spm",
                ],
                "input": "<tmp>/file.1",
                "output": "<tmp>/file.1.out",
                "log": "<tmp>/file.1.log",
                "devices": ["0", "1", "2", "3"],
                "workspace": "12000",
                "mini-batch-words": "4000",
                "precision": "float16",
                "models": "<src>/data/tests_data/test_translate/fake-model.npz",
            },
            # extra_args
            None,
        ),
        (
            # task
            "backtranslations-mono-trg-translate",
            # marian_args
            {
                "beam-size": "12",
                "config": "<src>/pipeline/translate/decoder.yml",
                "vocabs": [
                    "<src>/data/tests_data/test_translate/vocab.en.spm",
                    "<src>/data/tests_data/test_translate/vocab.ru.spm",
                ],
                "input": "<tmp>/file.1",
                "output": "<tmp>/file.1.out",
                "log": "<tmp>/file.1.log",
                "devices": ["0", "1", "2", "3"],
                "workspace": "12000",
                "mini-batch-words": "2000",
                "models": "<src>/data/tests_data/test_translate/fake-model.npz",
            },
            # extra_args
            None,
        ),
        (
            # task
            "backtranslations-mono-trg-translate",
            # marian_args
            {
                "beam-size": "1",
                "config": "<src>/pipeline/translate/decoder.yml",
                "vocabs": [
                    "<src>/data/tests_data/test_translate/vocab.en.spm",
                    "<src>/data/tests_data/test_translate/vocab.ru.spm",
                ],
                "input": "<tmp>/file.1",
                "output": "<tmp>/file.1.out",
                "log": "<tmp>/file.1.log",
                "devices": ["0", "1", "2", "3"],
                "workspace": "12000",
                "mini-batch-words": "2000",
                "models": "<src>/data/tests_data/test_translate/fake-model.npz",
                "output-sampling": ["topk", "10"],
            },
            # extra_args
            # A real marian-decoder process would fail if beam-size is provided 2 times
            # in this test, the argument is repeated because marian-args has it
            # and we provide an additional in this extra_args
            # however this won't happen in prod because the output-sampling will be enabled
            # in the pipeline conf, together with the beam-size
            # here we have to specify beam 1 because the default beam 12 taken
            # from the pipeline config for the tests fails with sampling enabled
            ["--beam-size", "1", "--output-sampling", "[topk,", "10]"],
        ),
    ],
    ids=lambda params: params[0],
)
def test_translate_mono(params: tuple[str, dict, list], data_dir: DataDir):
    task, marian_args, extra_args = params
    data_dir.create_zst("file.1.zst", en_sample)
    data_dir.create_file("fake-model.npz", "")
    data_dir.print_tree()
    data_dir.run_task(
        f"{task}-en-ru-1/10",
        env={
            "MARIAN": str(fixtures_path),
            "TEST_ARTIFACTS": data_dir.path,
            "USE_CPU": "1",
        },
        extra_args=extra_args,
    )
    data_dir.print_tree()

    assert (
        data_dir.read_text("artifacts/file.1.out.zst") == en_sample.upper()
    ), "The text is pseudo-translated"

    args = json.loads(data_dir.read_text("marian-decoder.args.txt"))
    assert sanitize_marian_args(args) == marian_args
