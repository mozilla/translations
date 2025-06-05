from pathlib import Path

import pytest
from fixtures import DataDir

from pipeline.common.downloads import read_lines

# With 47 lines, there will be 5 lines per file, except the last file which should have 2.
corpus_line_count = 47


@pytest.mark.parametrize(
    "task", ["distillation-mono-src-chunk-en", "backtranslations-mono-trg-chunk-ru"]
)
def test_split_mono(task: str):
    locale = task.split("-").pop()
    data_dir = DataDir("test_split_mono")
    data_dir.create_zst(
        f"mono.{locale}.zst", "\n".join([str(i) for i in range(corpus_line_count)]) + "\n"
    )
    data_dir.run_task(task)
    data_dir.print_tree()

    for i in range(10):
        Path(data_dir.join(f"artifacts/file.{i+1}.zst")).exists()

    with read_lines(data_dir.join("artifacts/file.9.zst")) as lines:
        assert list(lines) == ["40\n", "41\n", "42\n", "43\n", "44\n"]

    with read_lines(data_dir.join("artifacts/file.10.zst")) as lines:
        assert list(lines) == ["45\n", "46\n"], "The last file has a partial chunk"


def test_split_corpus():
    data_dir = DataDir("test_split_corpus")
    data_dir.mkdir("fetches")
    data_dir.create_zst(
        "fetches/corpus.en.zst", "\n".join([f"en-{i}" for i in range(corpus_line_count)]) + "\n"
    )
    data_dir.create_zst(
        "fetches/corpus.ru.zst", "\n".join([f"ru-{i}" for i in range(corpus_line_count)]) + "\n"
    )
    data_dir.run_task("distillation-parallel-src-chunk-en-ru")
    data_dir.print_tree()

    for i in range(10):
        Path(data_dir.join(f"artifacts/file.{i + 1}.zst")).exists()

    with read_lines(data_dir.join("artifacts/file.9.zst")) as lines:
        assert list(lines) == ["en-40\n", "en-41\n", "en-42\n", "en-43\n", "en-44\n"]

    with read_lines(data_dir.join("artifacts/file.10.zst")) as lines:
        assert list(lines) == ["en-45\n", "en-46\n"], "The last file has a partial chunk"

    with read_lines(data_dir.join("artifacts/file.9.ref.zst")) as lines:
        assert list(lines) == ["ru-40\n", "ru-41\n", "ru-42\n", "ru-43\n", "ru-44\n"]

    with read_lines(data_dir.join("artifacts/file.10.ref.zst")) as lines:
        assert list(lines) == ["ru-45\n", "ru-46\n"], "The last file has a partial chunk"
