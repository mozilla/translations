import json
import pytest
from fixtures import DataDir

from pipeline.common.downloads import read_lines

ada = [
    ("ADA 1", "АДА 1"),
    ("ADA 2", "АДА 2"),
    ("ADA 3", "АДА 3"),
    ("SHARED 1", "ШАРЕД 1"),
    ("SHARED 2", "ШАРЕД 2"),
    ("ADA 4", "АДА 4"),
    ("ADA 5", "АДА 5"),
]
wiki = [
    ("WIKI 1", "WИКИ 1"),
    ("WIKI 2", "WИКИ 2"),
    ("SHARED 3", "ШАРЕД 3"),
    ("SHARED 4", "ШАРЕД 4"),
    ("WIKI 3", "WИКИ 3"),
    ("SHARED 1", "ШАРЕД 1"),
    ("WIKI 4", "WИКИ 4"),
]
web_acquired = [
    ("WEB_ACQUIRED 1", "WЕБ_АЦQУИРЕД 1"),
    ("WEB_ACQUIRED 2", "WЕБ_АЦQУИРЕД 2"),
    ("SHARED 3", "ШАРЕД 3"),
    ("SHARED 4", "ШАРЕД 4"),
    ("WEB_ACQUIRED 3", "WЕБ_АЦQУИРЕД 3"),
    ("SHARED 2", "ШАРЕД 2"),
    ("WEB_ACQUIRED 4", "WЕБ_АЦQУИРЕД 4"),
]


def build_dataset_contents(lines: list[tuple[str, str]], index):
    return "\n".join([line[index] for line in lines]) + "\n"


@pytest.fixture(scope="function")
def data_dir():
    data_dir = DataDir("test_merge_corpus")
    data_dir.mkdir("artifacts")
    data_dir.create_zst("ada83_v1.en.zst", build_dataset_contents(ada, 0))
    data_dir.create_zst("ada83_v1.ru.zst", build_dataset_contents(ada, 1))
    data_dir.create_zst("ELRC-3075-wikipedia_health_v1.en.zst", build_dataset_contents(wiki, 0))
    data_dir.create_zst("ELRC-3075-wikipedia_health_v1.ru.zst", build_dataset_contents(wiki, 1))
    data_dir.create_zst("ELRC-web_acquired_data.en.zst", build_dataset_contents(web_acquired, 0))
    data_dir.create_zst("ELRC-web_acquired_data.ru.zst", build_dataset_contents(web_acquired, 1))
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


@pytest.mark.parametrize(
    "name",
    ["corpus", "devset"],
)
def test_merge_corpus(data_dir: DataDir, name):
    data_dir.run_task(
        # Tasks merge-corpus-en-ru, and merge-devset-en-ru.
        f"merge-{name}-en-ru",
    )
    data_dir.print_tree()
    assert_dataset(
        data_dir,
        f"artifacts/{name}.en.zst",
        sorted_lines=[
            "ADA 1\n",
            "ADA 2\n",
            "ADA 3\n",
            "ADA 4\n",
            "ADA 5\n",
            "SHARED 1\n",
            "SHARED 2\n",
            "SHARED 3\n",
            "SHARED 4\n",
            "WEB_ACQUIRED 1\n",
            "WEB_ACQUIRED 2\n",
            "WEB_ACQUIRED 3\n",
            "WEB_ACQUIRED 4\n",
            "WIKI 1\n",
            "WIKI 2\n",
            "WIKI 3\n",
            "WIKI 4\n",
        ],
    )

    assert_dataset(
        data_dir,
        f"artifacts/{name}.ru.zst",
        sorted_lines=[
            "WЕБ_АЦQУИРЕД 1\n",
            "WЕБ_АЦQУИРЕД 2\n",
            "WЕБ_АЦQУИРЕД 3\n",
            "WЕБ_АЦQУИРЕД 4\n",
            "WИКИ 1\n",
            "WИКИ 2\n",
            "WИКИ 3\n",
            "WИКИ 4\n",
            "АДА 1\n",
            "АДА 2\n",
            "АДА 3\n",
            "АДА 4\n",
            "АДА 5\n",
            "ШАРЕД 1\n",
            "ШАРЕД 2\n",
            "ШАРЕД 3\n",
            "ШАРЕД 4\n",
        ],
    )

    assert json.loads(data_dir.read_text(f"artifacts/{name}.stats.json")) == {
        "parallel_corpus": {
            "description": "The parallel corpora are merged and deduplicated",
            "filtered": 4,
            "kept": 17,
            "visited": 21,
        },
        "final_truncated": {
            "description": "The final result can be truncated by max_lines",
            "filtered": 0,
            "kept": 17,
            "visited": 17,
        },
        "datasets": [
            {
                "description": "ELRC-3075-wikipedia_health_v1",
                "filtered": 0,
                "kept": 7,
                "visited": 7,
            },
            {"description": "ELRC-web_acquired_data", "filtered": 2, "kept": 5, "visited": 7},
            {"description": "ada83_v1", "filtered": 2, "kept": 5, "visited": 7},
        ],
    }


@pytest.mark.parametrize(
    "name",
    ["corpus", "devset"],
)
def test_merge_devset_trimmed(data_dir: DataDir, name: str):
    data_dir.run_task(
        # Tasks merge-corpus-en-ru, and merge-devset-en-ru.
        f"merge-{name}-en-ru",
        # Replace the max_sentences.
        replace_args=[("None", "10")],
    )
    data_dir.print_tree()
    assert_dataset(
        data_dir,
        f"artifacts/{name}.en.zst",
        sorted_lines=[
            "ADA 1\n",
            "ADA 3\n",
            "ADA 4\n",
            "ADA 5\n",
            "SHARED 1\n",
            "SHARED 2\n",
            "SHARED 3\n",
            "WIKI 1\n",
            "WIKI 2\n",
            "WIKI 4\n",
        ],
    )

    assert_dataset(
        data_dir,
        f"artifacts/{name}.ru.zst",
        sorted_lines=[
            "WИКИ 1\n",
            "WИКИ 2\n",
            "WИКИ 4\n",
            "АДА 1\n",
            "АДА 3\n",
            "АДА 4\n",
            "АДА 5\n",
            "ШАРЕД 1\n",
            "ШАРЕД 2\n",
            "ШАРЕД 3\n",
        ],
    )

    assert json.loads(data_dir.read_text(f"artifacts/{name}.stats.json")) == {
        "parallel_corpus": {
            "description": "The parallel corpora are merged and deduplicated",
            "filtered": 4,
            "kept": 17,
            "visited": 21,
        },
        "final_truncated": {
            "description": "The final result can be truncated by max_lines",
            "filtered": 7,
            "kept": 10,
            "visited": 17,
        },
        "datasets": [
            {
                "description": "ELRC-3075-wikipedia_health_v1",
                "filtered": 0,
                "kept": 7,
                "visited": 7,
            },
            {"description": "ELRC-web_acquired_data", "filtered": 2, "kept": 5, "visited": 7},
            {"description": "ada83_v1", "filtered": 2, "kept": 5, "visited": 7},
        ],
    }
