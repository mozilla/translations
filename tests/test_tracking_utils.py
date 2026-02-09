import pytest
from fixtures import get_taskgraph_files
from tracking.translations_parser.utils import (
    ParsedTaskLabel,
    build_task_name,
    parse_task_label,
    parse_gcp_metric,
)


@pytest.mark.parametrize(
    "task_label, parsed_values",
    [
        (
            "evaluate-teacher-flores-flores_aug-title_devtest-lt-en-1_2",
            ("teacher-1", "flores", "devtest", "aug-title"),
        ),
        (
            "evaluate-finetune-teacher-sacrebleu-wmt19-lt-en-2_2",
            ("finetune-teacher-2", "sacrebleu", "wmt19", None),
        ),
        (
            "evaluate-student-sacrebleu-wmt19-lt-en",
            ("student", "sacrebleu", "wmt19", None),
        ),
        (
            "train-student-en-hu",
            ("student", None, None, None),
        ),
        (
            "eval_student-finetuned_flores_devtest",
            ("student-finetune", "flores", "devtest", None),
        ),
        (
            "eval_teacher-base0_flores_devtest",
            ("teacher-base-0", "flores", "devtest", None),
        ),
        (
            "train-backwards-en-ca",
            ("backwards", None, None, None),
        ),
        (
            "evaluate-teacher-flores-flores_dev-en-ca-1/2",
            ("teacher-1", "flores", "dev", None),
        ),
        (
            "train-teacher-ensemble",
            ("teacher-ensemble", None, None, None),
        ),
        (
            "evaluate-teacher-flores-flores_dev-en-ca",
            ("teacher-1", "flores", "dev", None),
        ),
        (
            "train-finetune-student-ru-en",
            ("finetune-student", None, None, None),
        ),
        (
            "train-teacher-ru-en-1",
            ("teacher-1", None, None, None),
        ),
        (
            "evaluate-backward-url-gcp_pytest-dataset_a0017e-en-ru",
            ("backwards", "url", "gcp_pytest-dataset_a0017e", None),
        ),
        (
            "train-teacher-ast-en-1",
            ("teacher-1", None, None, None),
        ),
        (
            # Test the 3-letter language codes like "Asturian".
            "evaluate-student-sacrebleu-wmt19-ast-en",
            ("student", "sacrebleu", "wmt19", None),
        ),
        (
            "evaluate-teacher-flores-devtest-ru-en-1",
            ("teacher-1", "flores", "devtest", None),
        ),
        (
            "distillation-student-model-train-en-hu",
            ("student", None, None, None),
        ),
        (
            "backtranslations-train-backwards-model-en-ca",
            ("backwards", None, None, None),
        ),
        (
            "train-teacher-model-ru-en-1",
            ("teacher-1", None, None, None),
        ),
        (
            "train-teacher-model-ru-en-2",
            ("teacher-2", None, None, None),
        ),
        (
            "train-backwards-en-ca",
            ("backwards", None, None, None),
        ),
        (
            "distillation-student-model-finetune-ru-en",
            ("finetune-student", None, None, None),
        ),
        (
            "backtranslations-train-backwards-model-zh_hant-en",
            ("backwards", None, None, None),
        ),
        (
            "evaluate-backward-url-gcp_pytest-dataset_a0017e-en-zh_hant",
            ("backwards", "url", "gcp_pytest-dataset_a0017e", None),
        ),
    ],
)
def test_parse_task_label(task_label, parsed_values):
    assert parse_task_label(task_label) == ParsedTaskLabel(*parsed_values)


def test_parse_labels_on_full_taskgraph():
    """Ensure that all the taskgraph task labels parse."""
    task_graph = get_taskgraph_files().full
    evaluate_tasks = [task for task in task_graph if task.startswith("evaluate-")]
    backwards = [
        task for task in task_graph if task.startswith("backtranslations-train-backwards-model")
    ]
    teacher = [task for task in task_graph if task.startswith("train-teacher-model")]
    student = [task for task in task_graph if task.startswith("distillation-student-model-train")]
    student_finetuned = [
        task for task in task_graph if task.startswith("distillation-student-model-finetune")
    ]

    # Ensure at least one task was found for each search.
    assert evaluate_tasks, "evaluate_tasks found"
    assert backwards, "backwards found"
    assert teacher, "teacher found"
    assert student, "student found"
    assert student_finetuned, "student_finetuned found"

    for task in [*evaluate_tasks, *backwards, *teacher, *student, *student_finetuned]:
        if (
            task.startswith("evaluate-")
            or task.startswith("backtranslations-train-backwards-model")
            or task.startswith("distillation-student-model-finetune")
            or task.startswith("train-teacher-model-model")
            or task.startswith("backtranslations-train-backwards-model")
        ):
            print(task)
            # This throws when it fails to parse.
            parse_task_label(task)


@pytest.mark.parametrize(
    "task_tags, values",
    [
        (
            {
                "os": "linux",
                "kind": "train-student",
                "label": "train-student-lt-en",
            },
            ("train", "student"),
        ),
        (
            {
                "os": "linux",
                "kind": "evaluate",
                "label": "evaluate-teacher-sacrebleu-sacrebleu_aug-upper_wmt19-lt-en-2/2",
            },
            ("evaluate", "teacher-2"),
        ),
    ],
)
def test_build_task_name(task_tags, values):
    task = {"tags": task_tags}
    assert build_task_name(task) == values


@pytest.mark.parametrize(
    "filename, parsed_values",
    [
        (
            "flores_aug-mix_devtest",
            ("flores", "aug-mix", "devtest"),
        ),
        (
            "flores_aug-title_devtest",
            ("flores", "aug-title", "devtest"),
        ),
        (
            "flores_aug-title-strict_devtest",
            ("flores", "aug-title-strict", "devtest"),
        ),
        (
            "flores_aug-typos_devtest",
            ("flores", "aug-typos", "devtest"),
        ),
        (
            "flores_aug-upper_devtest",
            ("flores", "aug-upper", "devtest"),
        ),
        (
            "flores_aug-upper-strict_devtest",
            ("flores", "aug-upper-strict", "devtest"),
        ),
        (
            "flores_devtest",
            ("flores", None, "devtest"),
        ),
        (
            "sacrebleu_aug-mix_wmt19",
            ("sacrebleu", "aug-mix", "wmt19"),
        ),
        (
            "sacrebleu_aug-title-strict_wmt19",
            ("sacrebleu", "aug-title-strict", "wmt19"),
        ),
        (
            "sacrebleu_aug-title_wmt19",
            ("sacrebleu", "aug-title", "wmt19"),
        ),
        (
            "sacrebleu_aug-typos_wmt19",
            ("sacrebleu", "aug-typos", "wmt19"),
        ),
        (
            "sacrebleu_aug-upper-strict_wmt19",
            ("sacrebleu", "aug-upper-strict", "wmt19"),
        ),
        (
            "sacrebleu_wmt19",
            ("sacrebleu", None, "wmt19"),
        ),
    ],
)
def test_gcp_metric(filename, parsed_values):
    assert tuple(parse_gcp_metric(filename)) == parsed_values


@pytest.mark.parametrize("filename", ["devtest", "tc_Tatoeba-Challenge-v2021-08-07", "test"])
def test_wrong_gcp_metric(filename):
    with pytest.raises(ValueError):
        parse_gcp_metric(filename)
