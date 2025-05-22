import pytest
from fixtures import DataDir

from pipeline.common.downloads import read_lines


def build_dataset_contents(name: str, line_count: int):
    return "\n".join([f"{name} - {index}" for index in range(line_count)]) + "\n"


@pytest.fixture(scope="function")
def data_dir():
    data_dir = DataDir("test_corpus_analyze_parallel_sample")
    data_dir.mkdir("artifacts")
    data_dir.create_zst("ada83_v1.en.zst", build_dataset_contents("ada sentence", 3000))
    data_dir.create_zst("ada83_v1.ru.zst", build_dataset_contents("предложение ада", 3000))
    data_dir.create_zst(
        "ELRC-3075-wikipedia_health_v1.en.zst",
        build_dataset_contents("wikipedia health sentence", 2000),
    )
    data_dir.create_zst(
        "ELRC-3075-wikipedia_health_v1.ru.zst",
        build_dataset_contents("википедия приговор о здоровье", 2000),
    )
    data_dir.create_zst(
        "ELRC-web_acquired_data.en.zst", build_dataset_contents("web acquired data", 300)
    )
    data_dir.create_zst(
        "ELRC-web_acquired_data.ru.zst",
        build_dataset_contents("данные, полученные через Интернет", 300),
    )
    return data_dir


def assert_dataset(data_dir: DataDir, path: str, sorted_lines: list[str]):
    with read_lines(data_dir.join(path)) as lines_iter:
        # Sort the dataset, as the sorted lines are easier to scan and reason with.
        # The datasets is still checked to be shuffled by comparing the original
        # and sorted lines.
        corpus_lines = list(lines_iter)
        corpus_lines_sorted = list(corpus_lines)
        corpus_lines_sorted.sort()

        assert corpus_lines_sorted == sorted_lines
        assert corpus_lines != corpus_lines_sorted, "The results are shuffled."


def test_corpus_analyze_parallel_sample(data_dir: DataDir):
    data_dir.run_task("corpus-analyze-parallel-sample-en-ru")
    data_dir.print_tree()
    data_dir.assert_files(
        [
            "artifacts/ada83_v1.sample.txt",
            "artifacts/corpus.sample.100.txt",
            "artifacts/corpus.sample.15.txt",
            "artifacts/ELRC-3075-wikipedia_health_v1.sample.txt",
            "artifacts/ELRC-web_acquired_data.sample.txt",
            # The original fetches are just left behind.
            "ada83_v1.en.zst",
            "ada83_v1.ru.zst",
            "ELRC-3075-wikipedia_health_v1.en.zst",
            "ELRC-3075-wikipedia_health_v1.ru.zst",
            "ELRC-web_acquired_data.en.zst",
            "ELRC-web_acquired_data.ru.zst",
        ]
    )

    byte_order_mark = "\ufeff"
    sample_lines = data_dir.read_text("artifacts/ada83_v1.sample.txt").split("\n")
    assert sample_lines[:9] == [
        byte_order_mark + "en: ada sentence - 2111",
        "ru: предложение ада - 2111",
        "",
        "en: ada sentence - 893",
        "ru: предложение ада - 893",
        "",
        "en: ada sentence - 1565",
        "ru: предложение ада - 1565",
        "",
    ]
    assert len(sample_lines), 1000

    assert data_dir.read_text("artifacts/corpus.sample.15.txt").split("\n")[:15] == [
        byte_order_mark + "--------",
        "Dataset: ELRC-3075-wikipedia_health_v1",
        "Lines: 2000",
        "Sample: 15",
        "--------",
        "",
        "en: wikipedia health sentence - 1754",
        "ru: википедия приговор о здоровье - 1754",
        "",
        "en: wikipedia health sentence - 1897",
        "ru: википедия приговор о здоровье - 1897",
        "",
        "en: wikipedia health sentence - 843",
        "ru: википедия приговор о здоровье - 843",
        "",
    ]
