import logging
import os
import re
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional

import yaml

import taskcluster
from taskcluster.download import downloadArtifactToFile

logger = logging.getLogger(__name__)

# Keywords used to split eval filenames into model and dataset
DATASET_KEYWORDS = ["flores", "mtdata", "sacrebleu"]

# Tags usually ends with project (e.g. `en-nl` or `eng-nld`)
TAG_PROJECT_SUFFIX_REGEX = re.compile(r"((-\w{2}){2}|(-\w{3}){2})$")

MULTIPLE_TRAIN_SUFFIX = re.compile(r"(-\d+)/\d+$")

MODEL_TASK_NAMES = "|".join(
    [
        "backtranslations-train-backwards-model",
        "backward",
        "backwards",
        "distillation-student-model-train",
        "distillation-student-model-finetune",
        "finetune-student",
        "finetune-teacher",
        "finetuned-student",
        "quantized",
        "student-finetuned",
        "student",
        "teacher-all",
        "teacher-base",
        "teacher-ensemble",
        "teacher-finetuned",
        "teacher-model",
        "teacher-parallel",
        "teacher",
    ]
)
EVAL_TASK_NAMES = "|".join(
    [
        "backward",
        "backwards",
        "finetune-student",
        "finetune-teacher",
        "finetuned-student",
        "quantized",
        "student-finetuned",
        "student",
        "teacher-all",
        "teacher-base",
        "teacher-ensemble",
        "teacher-finetuned",
        "teacher-parallel",
        "teacher",
    ]
)

# This regex needs to work on historic runs as well as the current tasks.
TRAIN_LABEL_REGEX = re.compile(
    # The "train-" prefix is optional.
    r"^"
    r"(train-)?"
    #
    # Capture what model is being run, for instance:
    #   train-teacher-ru-en-1
    #         ^^^^^^^
    r"(?P<model>"
    f"({MODEL_TASK_NAMES})"
    r")"
    #
    # Capture some legacy numeric suffixes.
    r"(-?(?P<suffix>\d+))?"
    r"[_-]?"
    #
    # Match the languages. BCP 47 language tags can be 2 or 3 letters long.
    #   train-teacher-model-ru-en-1
    #                 ^^ ^^
    r"(?P<lang>[a-z]{2,3}-[a-z]{2,3})?"
    r"-?"
    #
    # Match the task chunking, for instance:
    #   train-teacher-model-ru-en-1
    #                       ^
    # Legacy pattern:
    #   train-teacher-model-ru-en-1/3
    #                       ^
    r"-?((?P<task_suffix>\d+)((\/|_)\d+)?)?"
    #
    r"$"
)
EVAL_REGEX = re.compile(
    r"^"
    # Match evaluate steps.
    r"(evaluate|eval)[-_]"
    #
    # Capture what model is being run, for instance:
    #   evaluate-student-sacrebleu-wmt19-lt-en
    #            ^^^^^^^
    r"(?P<model>"
    f"({EVAL_TASK_NAMES})"
    r")"
    #
    # Capture some legacy numeric suffixes.
    r"(-?(?P<suffix>\d+))?"
    r"[_-]"
    #
    # Capture which importer is being used.
    #   evaluate-teacher-flores-flores_aug-title_devtest-lt-en-1_2
    #                    ^^^^^^
    r"(?P<importer>flores|mtdata|sacrebleu|url|tmx)"
    r"(?P<extra_importer>-flores|-mtdata|-sacrebleu)?"
    r"[_-]"
    #
    # Capture any augmentations
    #   evaluate-teacher-flores-flores_aug-title_devtest-lt-en-1_2
    #                                  ^^^^^^^^^
    r"(?P<aug>aug-[^_]+)?"
    #
    # Capture the dataset.
    #   evaluate-quantized-mtdata_aug-mix_Neulab-tedtalks_eng-lit-lt-en
    #                                     ^^^^^^^^^^^^^^^^^^^^^^^
    r"_?(?P<dataset>[-\w\d_]*?(-[a-z]{3}-[a-z]{3})?)?"
    #
    # Match language (project) suffix.
    # evaluate-teacher-flores-devtest-ru-en-1
    #                                 ^^^^^
    #
    r"-?(?P<lang>[a-z]{2,3}-[a-z]{2,3})?"
    #
    # Match the task chunking, for instance:
    #   evaluate-teacher-flores-flores_dev-en-ca-1/2
    #                                            ^
    #   evaluate-teacher-flores-flores_aug-title_devtest-lt-en-1_2
    #                                                          ^
    r"(-(?P<task_suffix>\d+)([\/|_]\d+)?)?"
    #
    r"$"
)

LABEL_PREFIX_LANGTAG_AND_ENSEMBLE_NUMBER = re.compile(
    r"^"
    # distillation-student-model-train
    r"(?P<label>\w+)"
    # An optional langtag, e.g. "-en-hu"
    r"(-(?P<lang>[a-z]{2,3}-[a-z]{2,3})?)?"
    # An optional model number for ensembles
    r"(?P<ensemble_number>-\d+)?"
    #
    r"$"
)

# The task labels describe where they are in the pipeline, and what task is being done.
# Map the extracted regex label part (the labels without the "train-" prefix, the langpair
# and the ensemble number) to the actual name of the model being produced such that it
# matches the historic runs and has simple consistent naming.
LABEL_PART_TO_MODEL_NAME = {
    "distillation-student-model-train": "student",
    "distillation-student-model-finetune": "finetune-student",
    "teacher-model": "teacher",
    "backtranslations-train-backwards-model": "backwards",
    # This one is for legacy task names:
    "backward": "backwards",
}

queue = taskcluster.Queue({"rootUrl": "https://firefox-ci-tc.services.mozilla.com"})


class ParsedTaskLabel(NamedTuple):
    model: str
    importer: Optional[str]
    dataset: Optional[str]
    augmentation: Optional[str]


class ParsedGCPMetric(NamedTuple):
    importer: str
    augmentation: Optional[str]
    dataset: Optional[str]


def patch_model_name(model, suffix=None):
    """Model Naming and suffix may be inconsistent between different sources"""
    if suffix is None:
        # Try to autodetect suffix based on name
        re_match = re.search(r"(?P<end>-?(?P<suffix>\d+))$", model)
        if re_match:
            re_match = re_match.groupdict()
            model = model[: -len(re_match["end"])]
            suffix = re_match["suffix"]

    model = model.replace("finetuned", "finetune")
    model = LABEL_PART_TO_MODEL_NAME.get(model, model)

    if not suffix and model == "teacher":
        # Keep the index on teacher runs for compatibility with legacy models
        # https://github.com/mozilla/translations/issues/573
        suffix = "1"
    if suffix:
        model = f"{model}-{suffix}"
    return model


def parse_task_label(task_label: str) -> ParsedTaskLabel:
    """
    Parse details out of train-* and evaluate-* task labels.
    """
    # First try to parse a simple training label
    match = TRAIN_LABEL_REGEX.match(task_label)
    if match is None:
        # Else try to parse an evaluation label with importer, dataset and auugmentation
        match = EVAL_REGEX.match(task_label)
    if not match:
        raise ValueError(f"Label could not be parsed: {task_label}")
    groups = match.groupdict()
    model = patch_model_name(
        groups["model"], suffix=groups.get("suffix") or groups.get("task_suffix")
    )

    return ParsedTaskLabel(model, groups.get("importer"), groups.get("dataset"), groups.get("aug"))


def taskcluster_log_filter(headers: Sequence[Sequence[str]]) -> bool:
    """
    Check TC log contain a valid task header i.e. ('task', <timestamp>)
    """
    for values in headers:
        if not values or len(values) != 2:
            continue
        base, timestamp = values
        if base != "task":
            continue
        try:
            datetime.fromisoformat(timestamp.rstrip("Z"))
            return True
        except ValueError:
            continue
    return False


def build_task_name(task: dict):
    """
    Build a simpler task name using a Taskcluster task payload (without status)
    """
    prefix = task["tags"]["kind"].split("-")[0]
    label_value = task.get("tags", {}).get("label", None)
    if label_value is None:
        raise ValueError("Task has no label")
    label = parse_task_label(label_value)
    return prefix, label.model


def metric_from_tc_context(chrf: float, bleu: float, comet: float):
    """
    Find the various names needed to build a metric directly from a Taskcluster task
    """
    from translations_parser.data import Metric

    task_id = os.environ.get("TASK_ID")
    if not task_id:
        raise Exception("Evaluation metric can only be build in taskcluster")

    # CI task groups do not expose any configuration, so we must use default values
    queue = taskcluster.Queue({"rootUrl": os.environ["TASKCLUSTER_PROXY_URL"]})
    task = queue.task(task_id)
    parsed = parse_task_label(task["tags"]["label"])

    # Multiply comet metric by 100 to match other metrics percentage style
    comet *= 100

    return Metric(
        importer=parsed.importer,
        dataset=parsed.dataset,
        augmentation=parsed.augmentation,
        chrf=chrf,
        bleu_detok=bleu,
        comet=comet,
    )


def publish_group_logs_from_tasks(
    *,
    project: str,
    group: str,
    suffix: str = "",
    metrics_tasks: dict[str, dict] = {},
    config: dict = {},
):
    """
    Publish a fake run, named 'group_logs' to Weight & Biases from a Taskcluster context.
    In case project or group is left to None, both values will be detected from Taskcluster.
    `metrics_tasks` optionally contains finished evaluation tasks that will be published as new runs.
    """
    from translations_parser.publishers import WandB

    message = "Handling group_logs publication"
    if metrics_tasks:
        message += f" with {len(metrics_tasks)} extra evaluation tasks"
    logger.info(message)

    with tempfile.TemporaryDirectory() as temp_dir:
        logs_folder = Path(temp_dir) / "logs"
        metrics_folder = logs_folder / project / group / "metrics"
        metrics_folder.mkdir(parents=True, exist_ok=True)

        # Group and publish remaining metrics tasks via the logs publication
        for metric_task_id, metrics_task in metrics_tasks.items():
            filename = metrics_task["task"]["tags"]["label"]
            if re_match := MULTIPLE_TRAIN_SUFFIX.search(filename):
                (train_suffix,) = re_match.groups()
                filename = MULTIPLE_TRAIN_SUFFIX.sub(train_suffix, filename)

            metric_artifact = next(
                (
                    artifact["name"]
                    for artifact in queue.listLatestArtifacts(metric_task_id)["artifacts"]
                    if artifact["name"].endswith(".metrics")
                ),
                None,
            )
            if metric_artifact is None:
                logger.error(f"No .metric artifact found for task {metric_task_id}, skipping.")
                continue

            with (metrics_folder / f"{filename}.metrics").open("wb") as log_file:
                downloadArtifactToFile(
                    log_file,
                    taskId=metrics_task["status"]["taskId"],
                    name=metric_artifact,
                    queueService=queue,
                )

        # Dump experiment config so it is published on group_logs
        config_path = Path(temp_dir) / "experiments" / project / group / "config.yml"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w") as config_file:
            yaml.dump(config, config_file)

        parents = str(logs_folder.resolve()).strip().split("/")
        WandB.publish_group_logs(
            logs_parent_folder=parents,
            project=project,
            group=group,
            suffix=suffix,
            existing_runs=[],
        )


def suffix_from_group(task_group_id: str) -> str:
    # Simply return the first 5 characters of the Taskcluster group ID as unique runs suffix
    assert (
        len(task_group_id) >= 5
    ), f"Taskcluster group ID should contain more than 5 characters: {task_group_id}"
    return f"_{task_group_id[:5]}"


def get_lines_count(file_path: str) -> int:
    with open(file_path, "r") as f:
        return sum(1 for _ in f)


def parse_gcp_metric(filename: str) -> tuple[str, str, str]:
    importer, *extra_str = filename.split("_", 1)
    if importer not in DATASET_KEYWORDS:
        raise ValueError(f"Importer {importer} is not supported")

    extra_args = {"dataset": None}
    if extra_str:
        (extra_str,) = extra_str
        re_match = re.match(
            r"(?P<augmentation>aug-[^_]+)?_?(?P<dataset>[-\w\d_]+(-[a-z]{3}-[a-z]{3})?)",
            extra_str,
        )
        if not re_match:
            raise ValueError(f"Could not detect augmentation nor dataset from {extra_str}")
        extra_args.update(re_match.groupdict())

    return ParsedGCPMetric(importer, **extra_args)
