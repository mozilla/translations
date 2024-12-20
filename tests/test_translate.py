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
    shutil.copyfile("tests/data/vocab.spm", data_dir.join("vocab.spm"))
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
        "translate-corpus-en-ru-1/10",
        env={
            "MARIAN": str(fixtures_path),
            "TEST_ARTIFACTS": data_dir.path,
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
            "<src>/data/tests_data/test_translate/vocab.spm",
            "<src>/data/tests_data/test_translate/vocab.spm",
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
        "translate-corpus-en-ru-1/10",
        env={
            "MARIAN": str(fixtures_path),
            "TEST_ARTIFACTS": data_dir.path,
        },
    )

    data_dir.print_tree()

    assert data_dir.read_text("artifacts/file.1.nbest.zst") == "", "The text is empty"


mono_args = {
    "src": {
        "config": "<src>/pipeline/translate/decoder.yml",
        "vocabs": [
            "<src>/data/tests_data/test_translate/vocab.spm",
            "<src>/data/tests_data/test_translate/vocab.spm",
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
    "trg": {
        "beam-size": "12",
        "config": "<src>/pipeline/translate/decoder.yml",
        "vocabs": [
            "<src>/data/tests_data/test_translate/vocab.spm",
            "<src>/data/tests_data/test_translate/vocab.spm",
        ],
        "input": "<tmp>/file.1",
        "output": "<tmp>/file.1.out",
        "log": "<tmp>/file.1.log",
        "devices": ["0", "1", "2", "3"],
        "workspace": "12000",
        "mini-batch-words": "2000",
        "models": "<src>/data/tests_data/test_translate/fake-model.npz",
    },
}


@pytest.mark.parametrize("direction", ["src", "trg"])
def test_translate_mono(direction: str, data_dir: DataDir):
    data_dir.create_zst("file.1.zst", en_sample)
    data_dir.create_file("fake-model.npz", "")
    data_dir.print_tree()
    data_dir.run_task(
        f"translate-mono-{direction}-en-ru-1/10",
        env={
            "MARIAN": str(fixtures_path),
            "TEST_ARTIFACTS": data_dir.path,
        },
    )
    data_dir.print_tree()

    assert (
        data_dir.read_text("artifacts/file.1.out.zst") == en_sample.upper()
    ), "The text is pseudo-translated"

    args = json.loads(data_dir.read_text("marian-decoder.args.txt"))
    assert sanitize_marian_args(args) == mono_args[direction]
