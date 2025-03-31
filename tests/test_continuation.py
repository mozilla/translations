from dataclasses import dataclass
import shutil
from typing import Any
import pytest
import yaml
import json
from pathlib import Path
from fixtures import DataDir, get_taskgraph_files


class CorporaMocks:
    """
    Provides all of the files and URLs for mocking out corpora.
    """

    def __init__(self, name: str):
        def build_corpus(name):
            return "\n".join([f"{name} {i}" for i in range(3)]) + "\n"

        self.name = name
        self.src = build_corpus(f"{name} ru")
        self.trg = build_corpus(f"{name} en")
        self.tok_src = build_corpus(f"{name} tok ru")
        self.tok_trg = build_corpus(f"{name} tok en")
        self.aln = build_corpus(f"{name} alignments")

    def get_fetch_mocks(self, data_dir: DataDir):
        mocks = {}
        downloads_path = "mocked-downloads/corpora"
        data_dir.mkdir(downloads_path)

        def add_mock(name, contents):
            url = f"https://example.com/{name}"
            mocks[url] = data_dir.create_zst(f"{downloads_path}/{name}", contents)

        add_mock(f"{self.name}.ru.zst", self.src)
        add_mock(f"{self.name}.en.zst", self.trg)
        add_mock(f"{self.name}.tok-icu.ru.zst", self.tok_src)
        add_mock(f"{self.name}.tok-icu.en.zst", self.tok_trg)
        add_mock(f"{self.name}.aln.zst", self.aln)

        return mocks


class VocabMock:
    """
    Provides all of the files and URLs for mocking out corpora.
    """

    def __init__(self):
        self.src = "vocab ru"
        self.trg = "vocab en"

    def get_fetch_mocks(self, data_dir: DataDir):
        mocks = {}
        downloads_path = "mocked-downloads/vocab"
        data_dir.mkdir(downloads_path)

        def add_mock(name, contents):
            url = f"https://example.com/{name}"
            mocks[url] = data_dir.create_file(f"{downloads_path}/{name}", contents)

        add_mock("vocab.spm", "vocab")
        # TODO - Split vocab
        # add_mock("vocab.ru.zst", self.src)
        # add_mock("vocab.en.zst", self.trg)

        return mocks


class ModelMocks:
    """
    Provides all of the files and URLs for mocking out corpora.
    """

    def __init__(self, name: str):
        self.name = name
        self.model = f"{name} model file"
        self.decoder = f"{name} decoder file"
        self.vocab = f"{name} vocab file"
        # TODO - split vocab
        # self.vocab_src = "vocab.ru.spm"
        # self.vocab_tr = "vocab.en.spm"

    def get_fetch_mocks(self, data_dir: DataDir):
        mocks = {}
        downloads_path = f"mocked-downloads/model-{self.name}"
        data_dir.mkdir(downloads_path)

        def add_mock(name, contents):
            url = f"https://example.com/ru-en/backwards/{name}"
            mocks[url] = data_dir.create_file(f"{downloads_path}/{name}", contents)

        add_mock("model.npz.best-chrf.npz", self.model)
        add_mock("model.npz.best-chrf.npz.decoder.yml", self.decoder)
        add_mock("vocab.spm", self.vocab)

        # TODO - split vocab
        # add_mock("vocab.ru.spm", self.vocab_src)
        # add_mock("vocab.en.spm", self.vocab_trg)

        return mocks


expected_artifacts_by_task_label = {
    "continuation-corpus-backtranslations-ru-en": [
        "mono.en.zst",
        "mono.ru.zst",
    ],
    "continuation-corpus-original-parallel-ru-en": [
        "corpus.en.zst",
        "corpus.ru.zst",
    ],
    "continuation-corpus-student-distillation-ru-en": [
        "corpus.en.zst",
        "corpus.ru.zst",
    ],
}


def get_config_rewriter(yaml_str: str):
    """Returns a function that will rewrite the config for corpus continuation."""

    def rewrite(config: dict[str, Any]):
        corpora_yaml = yaml.safe_load(yaml_str)
        config["datasets"] = {
            "devtest": config["datasets"]["devtest"],
            "test": config["datasets"]["test"],
        }
        config["continuation"] = corpora_yaml["continuation"]

    return rewrite


@dataclass
class Continuation:
    task_label: str
    files: list[str]


@dataclass
class TestParams:
    test_name: str
    config_yaml: str
    included_task_labels: set[str]
    excluded_task_labels: set[str]


continuation_artifacts = {
    "continuation-vocab": [
        "vocab.spm",
        # TODO - For split vocab - ["vocab.ru.spm", "vocab.en.spm"]
    ],
    "continuation-model-backwards": [
        "final.model.npz.best-chrf.npz",
        "final.model.npz.best-chrf.npz.decoder.yml",
        "vocab.spm",
        # TODO - For split vocab - ["vocab.ru.spm", "vocab.en.spm"]
    ],
    "continuation-model-teacher": [
        "final.model.npz.best-chrf.npz",
        "final.model.npz.best-chrf.npz.decoder.yml",
        "vocab.spm",
        # TODO - For split vocab - ["vocab.ru.spm", "vocab.en.spm"]
    ],
    "continuation-corpus-student-distillation": [
        "corpus.ru.zst",
        "corpus.en.zst",
    ],
}

test_params: list[TestParams] = [
    TestParams(
        test_name="teacher_no_alignments",
        config_yaml="""
            continuation:
                vocab:
                    src: https://example.com/vocab.spm
                    trg: https://example.com/vocab.spm
                models:
                    backwards:
                        url: https://example.com/ru-en/backwards
                        mode: use
                        type: default
                corpora:
                    backtranslations:
                        src: https://example.com/backtranslations.ru.zst
                        trg: https://example.com/backtranslations.en.zst
                    original-parallel:
                        src: https://example.com/original-parallel.ru.zst
                        trg: https://example.com/original-parallel.en.zst

        """,
        included_task_labels={
            "alignments-backtranslated-ru-en",
            "alignments-original-ru-en",
            "alignments-student-ru-en",
            "cefilter-ru-en",
            "continuation-vocab-ru-en",
            "continuation-model-backwards-ru-en",
            "continuation-corpus-backtranslations-ru-en",
            "continuation-corpus-original-parallel-ru-en",
            "merge-devset-ru-en",
            "train-student-ru-en",
            "train-teacher-ru-en-1",
        },
        excluded_task_labels={
            "continuation-corpus-student-distillation-ru-en",
            "continuation-model-teacher-ru-en",
            "merge-corpus-ru-en",
            "merge-mono-src-ru",
            "merge-mono-trg-en",
            "train-backwards-ru-en",
            "train-vocab-ru-en",
        },
    ),
    TestParams(
        test_name="student_no_alignments",
        config_yaml="""
            continuation:
                vocab:
                    src: https://example.com/vocab.spm
                    trg: https://example.com/vocab.spm
                corpora:
                    student-distillation:
                        src: https://example.com/student-distillation.ru.zst
                        trg: https://example.com/student-distillation.en.zst
        """,
        included_task_labels={
            "alignments-student-ru-en",
            "continuation-corpus-student-distillation-ru-en",
            "continuation-vocab-ru-en",
            "merge-devset-ru-en",
            "train-student-ru-en",
        },
        excluded_task_labels={
            "alignments-backtranslated-ru-en",
            "alignments-original-ru-en",
            "cefilter-ru-en",
            "continuation-corpus-backtranslations-ru-en",
            "continuation-corpus-original-parallel-ru-en",
            "continuation-model-backwards-ru-en",
            "continuation-model-teacher-ru-en",
            "merge-corpus-ru-en",
            "merge-mono-src-ru",
            "merge-mono-trg-en",
            "train-backwards-ru-en",
            "train-teacher-ru-en-1",
            "train-vocab-ru-en",
        },
    ),
    TestParams(
        test_name="teacher_with_alignments",
        config_yaml="""
            continuation:
                vocab:
                    src: https://example.com/vocab.spm
                    trg: https://example.com/vocab.spm
                models:
                    backwards:
                        url: https://example.com/ru-en/backwards
                        mode: use
                        type: default
                corpora:
                    backtranslations:
                        src: https://example.com/backtranslations.ru.zst
                        trg: https://example.com/backtranslations.en.zst
                        tok-src: https://example.com/backtranslations.tok-icu.ru.zst
                        tok-trg: https://example.com/backtranslations.tok-icu.en.zst
                        alignments: https://example.com/backtranslations.aln.zst
                    original-parallel:
                        src: https://example.com/original-parallel.ru.zst
                        trg: https://example.com/original-parallel.en.zst
                        tok-src: https://example.com/original-parallel.tok-icu.ru.zst
                        tok-trg: https://example.com/original-parallel.tok-icu.en.zst
                        alignments: https://example.com/original-parallel.aln.zst
        """,
        included_task_labels={
            "cefilter-ru-en",
            "continuation-corpus-backtranslations-ru-en",
            "continuation-corpus-original-parallel-ru-en",
            "continuation-model-backwards-ru-en",
            "continuation-vocab-ru-en",
            "merge-devset-ru-en",
            "train-student-ru-en",
            "train-teacher-ru-en-1",
            "alignments-student-ru-en",
        },
        excluded_task_labels={
            "alignments-backtranslated-ru-en",
            "alignments-original-ru-en",
            "continuation-corpus-student-distillation-ru-en",
            "continuation-model-teacher-ru-en",
            "merge-mono-src-ru",
            "merge-mono-trg-en",
            "merge-corpus-ru-en",
            "train-backwards-ru-en",
            "train-vocab-ru-en",
        },
    ),
    TestParams(
        test_name="student_with_alignments",
        config_yaml="""
            continuation:
                vocab:
                    src: https://example.com/vocab.spm
                    trg: https://example.com/vocab.spm
                corpora:
                    student-distillation:
                        src: https://example.com/student-distillation.ru.zst
                        trg: https://example.com/student-distillation.en.zst
                        tok-src: https://example.com/student-distillation.tok-icu.ru.zst
                        tok-trg: https://example.com/student-distillation.tok-icu.en.zst
                        alignments: https://example.com/student-distillation.aln.zst
        """,
        included_task_labels={
            "continuation-corpus-student-distillation-ru-en",
            "continuation-vocab-ru-en",
            "merge-devset-ru-en",
            "train-student-ru-en",
        },
        excluded_task_labels={
            "cefilter-ru-en",
            "continuation-corpus-backtranslations-ru-en",
            "continuation-corpus-original-parallel-ru-en",
            "continuation-model-backwards-ru-en",
            "continuation-model-teacher-ru-en",
            "merge-corpus-ru-en",
            "alignments-backtranslated-ru-en",
            "alignments-original-ru-en",
            "alignments-student-ru-en",
            "train-backwards-ru-en",
            "train-teacher-ru-en-1",
            "train-vocab-ru-en",
        },
    ),
]


@pytest.mark.parametrize("params", test_params, ids=[p.test_name for p in test_params])
def test_continuation(params: TestParams):
    data_dir = DataDir(f"test_continuation_{params.test_name}")

    # Apply the continuation to the yaml.
    config_path = data_dir.rewrite_ci_config(get_config_rewriter(params.config_yaml))

    # Generate the taskgraph.
    tasks_by_id = get_taskgraph_files(config_path).resolved
    task_labels: list[str] = [task["label"] for task in tasks_by_id.values()]
    task_labels.sort()

    # Retain a copy of the task graph in the data dir to aid in debugging.
    task_graph_json = (Path(__file__).parent / "../artifacts/task-graph.json").resolve()
    artifacts_task_graph_json = data_dir.join("task-graph.json")
    shutil.copy(task_graph_json, artifacts_task_graph_json)
    print("The resolved tasks are available at:", artifacts_task_graph_json)

    print("Resolved tasks:")
    for task in tasks_by_id.values():
        print(" -", task["label"])
        for dependency_label in task["dependencies"].keys():
            print("    -", dependency_label)

    # Check that the tasks resolved correctly.
    missing_tasks = [
        task_label for task_label in params.included_task_labels if task_label not in task_labels
    ]
    assert missing_tasks == [], "All included tasks were resolved."

    extra_tasks = [
        task_label for task_label in params.excluded_task_labels if task_label in task_labels
    ]
    assert extra_tasks == [], "No excluded tasks were resolved."

    mocked_downloads = json.dumps(
        {
            **CorporaMocks("backtranslations").get_fetch_mocks(data_dir),
            **CorporaMocks("original-parallel").get_fetch_mocks(data_dir),
            **CorporaMocks("student-distillation").get_fetch_mocks(data_dir),
            **VocabMock().get_fetch_mocks(data_dir),
            **ModelMocks("backwards").get_fetch_mocks(data_dir),
            **ModelMocks("student").get_fetch_mocks(data_dir),
            **ModelMocks("teacher").get_fetch_mocks(data_dir),
        }
    )

    continuation_tasks = [
        task_label for task_label in task_labels if task_label.startswith("continuation")
    ]

    for continuation_task in continuation_tasks:
        # Ensure the artifacts are cleaned up from previous continuation tasks.
        artifacts_path = Path(data_dir.join("artifacts"))
        if artifacts_path.exists():
            shutil.rmtree(artifacts_path)
            artifacts_path.mkdir()

        # The task graph should be cached, and won't be regenerated. Clean the data_dir
        # before each continuation is run.
        data_dir.run_task(
            continuation_task,
            config=config_path,
            env={"MOCKED_DOWNLOADS": mocked_downloads},
        )
        data_dir.print_tree()
