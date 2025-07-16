import os
from pathlib import Path
import shutil

import pytest
import sh
from fixtures import (
    DataDir,
    TestParams,
    en_sample,
    get_config_rewriter,
    get_taskgraph_files,
    zh_sample,
    FIXTURES_PATH,
)

TRG = "ru"

SRC = "en"

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

# "|||" in the text can cause issues if joint fast_align style input is used
en_sample_with_separator = """The little girl, seeing she had lost one of her pretty shoes, grew angry, and said to the Witch, “Give me back my shoe!” ||| one
“I will not,” retorted the Witch, “for it is now my shoe, and not yours.”
“You are a wicked creature!” cried Dorothy. “You have no right to take my shoe from me.”
“I shall keep it, just the same,” said the Witch, laughing at her, “and someday I shall get the other one from you, too.”
This made Dorothy so very angry that she picked up the bucket of water that stood near and dashed it over the Witch, wetting her from head to foot.
Instantly the wicked woman gave a loud cry of fear, and then, as Dorothy looked at her in wonder, the Witch began to shrink and fall away.
“See what you have done!” she screamed. “In a minute I shall melt away.”
“I’m very sorry, indeed,” said Dorothy, who was truly frightened to see the Witch actually melting away like brown sugar before her very eyes.
"""

ru_sample_with_separator = """Маленькая девочка, увидев, что потеряла одну из своих красивых туфелек, рассердилась и сказала Ведьме: «Верни мне мою туфельку!» ||| один
«Я не буду, — парировала Ведьма, — потому что теперь это моя туфля, а не твоя».
«Ты злое существо!» - воскликнула Дороти. «Ты не имеешь права забирать у меня туфлю».
«Я все равно сохраню его, — сказала Ведьма, смеясь над ней, — и когда-нибудь я получу от тебя и другой».
Это так разозлило Дороти, что она взяла стоявшее рядом ведро с водой и облила им Ведьму, обмочив ее с головы до ног.
Мгновенно злая женщина громко вскрикнула от страха, а затем, когда Дороти с удивлением посмотрела на нее, Ведьма начала сжиматься и падать.
«Посмотри, что ты наделал!» она закричала. «Через минуту я растаю».
«Мне действительно очень жаль», — сказала Дороти, которая была по-настоящему напугана, увидев, что Ведьма тает, как коричневый сахар, у нее на глазах.
"""


def verify_alignments(data_dir, dataset, src, trg):
    aln_path = os.path.join(data_dir.path, "artifacts", f"{dataset}.aln.zst")
    assert os.path.exists(aln_path)

    sh.zstd("-d", aln_path)
    with open(aln_path[:-4], "r") as f:
        aln_lines = f.read().splitlines()

    src_tokenized_path = os.path.join(data_dir.path, "artifacts", f"{dataset}.tok-icu.{src}.zst")
    trg_tokenized_path = os.path.join(data_dir.path, "artifacts", f"{dataset}.tok-icu.{trg}.zst")

    sh.zstd("-d", src_tokenized_path, trg_tokenized_path)

    with open(src_tokenized_path[:-4], "r") as f:
        src_lines = f.read().splitlines()
    with open(trg_tokenized_path[:-4], "r") as f:
        trg_lines = f.read().splitlines()

    assert len(aln_lines) == len(src_lines)
    assert len(aln_lines) == len(trg_lines)

    # verify alignment indices
    with open(aln_path[:-4] + ".debug", "w") as f:
        for aln_line, src_line, trg_line in zip(aln_lines, src_lines, trg_lines):
            alns = [pair.split("-") for pair in aln_line.split()]
            src_tokens = src_line.split(" ")
            trg_tokens = trg_line.split(" ")
            src_tokens_num = len(src_tokens)
            trg_tokens_num = len(trg_tokens)

            assert all(
                int(src_idx) < src_tokens_num and int(trg_idx) < trg_tokens_num
                for src_idx, trg_idx in alns
            )

            aligned = []
            for src_idx, trg_idx in alns:
                aligned.append((src_tokens[int(src_idx)], trg_tokens[int(trg_idx)]))
            f.write(str(aligned))
            f.write("\n")


def test_teacher_original_alignments():
    data_dir = DataDir("test_alignments")
    data_dir.create_zst("corpus.en.zst", en_sample_with_separator)
    data_dir.create_zst("corpus.ru.zst", ru_sample_with_separator)
    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "SRC": SRC,
        "TRG": TRG,
        "ALN_CHUNK_LINES": "3",
    }

    data_dir.run_task("corpus-align-parallel-en-ru", env=env)
    data_dir.assert_files(
        [
            "artifacts/corpus.aln.zst",
            "artifacts/corpus.priors",
            "artifacts/corpus.tok-icu.en.zst",
            "artifacts/corpus.tok-icu.ru.zst",
        ]
    )

    verify_alignments(data_dir, "corpus", SRC, TRG)


def test_teacher_original_alignments_zh():
    data_dir = DataDir("test_alignments")
    data_dir.create_zst("corpus.en.zst", en_sample)
    data_dir.create_zst("corpus.zh.zst", zh_sample)
    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "SRC": "en",
        "TRG": "zh",
        "ALN_CHUNK_LINES": "3",
    }

    data_dir.run_task(
        "corpus-align-parallel-en-zh",
        env=env,
        config=os.path.abspath(os.path.join(FIXTURES_PATH, "config.pytest.enzh.yml")),
    )
    data_dir.assert_files(
        [
            "artifacts/corpus.aln.zst",
            "artifacts/corpus.priors",
            "artifacts/corpus.tok-icu.en.zst",
            "artifacts/corpus.tok-icu.zh.zst",
        ]
    )

    verify_alignments(data_dir, "corpus", "en", "zh")


def test_teacher_backtranslated_alignments():
    data_dir = DataDir("test_alignments")
    data_dir.create_zst("corpus.en.zst", en_sample_with_separator)
    data_dir.create_zst("mono.en.zst", en_sample_with_separator)
    data_dir.create_zst("corpus.ru.zst", ru_sample_with_separator)
    data_dir.create_zst("mono.ru.zst", ru_sample_with_separator)
    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "SRC": SRC,
        "TRG": TRG,
        "ALN_CHUNK_LINES": "3",
    }
    # get priors using the "original" task
    data_dir.run_task("corpus-align-parallel-en-ru", env=env)
    shutil.copyfile(
        os.path.join(data_dir.path, "artifacts", "corpus.priors"),
        os.path.join(data_dir.path, "corpus.priors"),
    )

    data_dir.run_task("corpus-align-backtranslations-en-ru", env=env)

    verify_alignments(data_dir, "mono", SRC, TRG)


def test_student_alignments():
    data_dir = DataDir("test_alignments")
    data_dir.create_zst("corpus.en.zst", en_sample_with_separator)
    data_dir.create_zst("corpus.ru.zst", ru_sample_with_separator)
    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "SRC": SRC,
        "TRG": TRG,
        "ALN_CHUNK_LINES": "3",
    }
    # get priors using the "original" task
    data_dir.run_task("corpus-align-parallel-en-ru", env=env)
    shutil.copyfile(
        os.path.join(data_dir.path, "artifacts", "corpus.priors"),
        os.path.join(data_dir.path, "corpus.priors"),
    )
    os.remove(os.path.join(data_dir.path, "artifacts", "corpus.aln.zst"))
    data_dir.create_zst("corpus.en.zst", en_sample_with_separator)
    data_dir.create_zst("corpus.ru.zst", ru_sample_with_separator)

    data_dir.run_task("corpus-align-distillation-en-ru", env=env)

    verify_alignments(data_dir, "corpus", SRC, TRG)


def test_distillation_corpus_shortlist():
    data_dir = DataDir("test_distillation_corpus_shortlist")
    data_dir.create_zst("corpus.en.zst", en_sample_with_separator)
    data_dir.create_zst("corpus.ru.zst", ru_sample_with_separator)
    env = {
        "TEST_ARTIFACTS": data_dir.path,
        "BIN": bin_dir,
        "MARIAN": marian_dir,
        "SRC": SRC,
        "TRG": TRG,
        "ALN_CHUNK_LINES": "3",
    }
    shutil.copyfile("tests/data/vocab.spm", os.path.join(data_dir.path, "vocab.en.spm"))
    shutil.copyfile("tests/data/vocab.spm", os.path.join(data_dir.path, "vocab.ru.spm"))

    data_dir.run_task("distillation-corpus-build-shortlist-en-ru", env=env)

    shortlist_path = os.path.join(data_dir.path, "artifacts", "lex.s2t.pruned.zst")
    assert os.path.exists(shortlist_path)


uploads_test_params: list[TestParams] = [
    TestParams(
        test_name="archive_corpora",
        config_yaml="""
            experiment:
                archive-corpora: true
        """,
        included_task_labels={
            "corpus-align-backtranslations-ru-en",
            "corpus-align-distillation-ru-en",
            "corpus-align-parallel-ru-en",
            "upload-artifacts-corpus-align-backtranslations-ru-en",
            "upload-artifacts-corpus-align-distillation-ru-en",
            "upload-artifacts-corpus-align-parallel-ru-en",
        },
        excluded_task_labels=set(),
    ),
    TestParams(
        test_name="no_archive_corpora",
        config_yaml="""
            experiment:
                archive-corpora: false
        """,
        included_task_labels={
            "corpus-align-backtranslations-ru-en",
            "corpus-align-distillation-ru-en",
            "corpus-align-parallel-ru-en",
        },
        excluded_task_labels={
            "upload-artifacts-corpus-align-backtranslations-ru-en",
            "upload-artifacts-corpus-align-distillation-ru-en",
            "upload-artifacts-corpus-align-parallel-ru-en",
        },
    ),
]


@pytest.mark.parametrize(
    "params", uploads_test_params, ids=[p.test_name for p in uploads_test_params]
)
def test_alignments_artifact_uploads(params: TestParams):
    """Ensure that alignments tasks have their artifacts uploaded only when
    `archive_corpora` is true."""
    data_dir = DataDir(f"test_alignments_{params.test_name}")

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
