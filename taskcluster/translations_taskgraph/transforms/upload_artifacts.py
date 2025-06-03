# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This transform is responsible for:
* Evaluating the `keyed-by` expressions for `upstream-artifacts` and `artifact-map`,
  which use a custom key of the kind of their upstream task. This allows us to
  specify different sets of and destinations for files that we upload.
* Substituting in the `step_dir` variable in `artifact-map`, which ensures we have
  an accurate and distinct directory to upload to for each task we upload artifacts
  for.
"""

import re

from taskgraph.transforms.base import TransformSequence
from taskgraph.util.schema import resolve_keyed_by

from translations_taskgraph.util.dependencies import get_upload_artifacts_upstream_dependency
from translations_taskgraph.util.substitution import substitute

transforms = TransformSequence()


@transforms.add
def evaluate_keyed_by(_, jobs):
    for job in jobs:
        upstream_kind = get_upload_artifacts_upstream_dependency(job["dependencies"])[0]
        for field in ("upstream-artifacts", "artifact-map"):
            resolve_keyed_by(
                job["worker"],
                field,
                item_name=job["description"],
                **{
                    "upstream-kind": upstream_kind,
                },
            )

        yield job


@transforms.add
def substitute_step_dir(_, jobs):
    for job in jobs:
        upstream_task = get_upload_artifacts_upstream_dependency(job["dependencies"])[1]
        step_dir = task_to_step_dir(upstream_task)
        substitute(job["worker"]["artifact-map"], step_dir=step_dir)
        yield job


# tasks for where the full task label is needed to avoid overwriting logs
# from different, eg: datasets
FULL_TASK_LABEL_STEP_DIR_TASKS = {
    "corpus-analyze-mono",
    "corpus-analyze-parallel",
    "corpus-clean-mono",
    "corpus-clean-parallel-bicleaner-ai",
    "corpus-clean-parallel",
}
TASKS_WITH_MATCHING_STEP_DIR_PREFIXES = {
    "backtranslations-mono-trg-chunk",
    "backtranslations-mono-trg-dechunk-translations",
    "corpus-align-backtranslations",
    "corpus-align-distillation",
    "corpus-align-parallel",
    "corpus-clean-parallel-fetch-bicleaner-model",
    "corpus-merge-devset",
    "corpus-merge-distillation",
    "corpus-merge-mono-src",
    "corpus-merge-mono-trg",
    "corpus-merge-parallel",
    "distillation-corpus-build-shortlist",
    "distillation-corpus-final-filtering",
    "distillation-mono-src-chunk",
    "distillation-mono-src-dechunk-translations",
    "distillation-parallel-src-chunk",
    "distillation-parallel-src-dechunk-translations",
    "distillation-parallel-src-translations-score",
}
CHUNKED_TASK_PREFIXES = {
    "backtranslations-mono-trg-translate",
    "distillation-mono-src-translate",
    "distillation-parallel-src-extract-best",
    "distillation-parallel-src-translate",
}


def task_to_step_dir(task_label: str) -> str:
    # Process these first, because we want to prioritize `corpus-clean-parallel-fetch-bicleaner-model`
    # not having SRC-TRG in its step directory.
    if prefixes := [
        prefix for prefix in TASKS_WITH_MATCHING_STEP_DIR_PREFIXES if task_label.startswith(prefix)
    ]:
        # there should only be only be one match!
        assert len(prefixes) == 1
        return prefixes[0]
    elif any([task_label.startswith(prefix) for prefix in FULL_TASK_LABEL_STEP_DIR_TASKS]):
        return task_label
    elif prefixes := [prefix for prefix in CHUNKED_TASK_PREFIXES if task_label.startswith(prefix)]:
        # there should only be only be one match!
        assert len(prefixes) == 1
        prefix = prefixes[0]
        n = get_chunk_number(task_label)
        return f"{prefix}{n}"
    elif task_label.startswith("backtranslations-train-backwards-model"):
        return "backward"
    elif task_label.startswith("build-vocab"):
        return "vocab"
    elif task_label.startswith("corpus-clean-parallel-fetch-bicleaner-model"):
        return "corpus-clean-parallel-fetch-bicleaner-model"
    elif task_label.startswith("distillation-student-model-finetune"):
        return "student-finetuned"
    elif task_label.startswith("distillation-student-model-quantize"):
        return "quantized"
    elif task_label.startswith("distillation-student-model-train"):
        return "student"
    elif task_label.startswith("evaluate-backward"):
        return "evaluation/backward"
    elif task_label.startswith("evaluate-finetuned-student"):
        return "evaluation/student-finetuned"
    elif task_label.startswith("evaluate-quantized"):
        return "evaluation/quantized"
    elif task_label.startswith("evaluate-student"):
        return "evaluation/student"
    elif task_label.startswith("evaluate-teacher-ensemble"):
        return "evaluation/teacher-ensemble"
    elif task_label.startswith("evaluate-teacher"):
        # the last character of these tasks is the teacher number
        # assumption: we will not have a double digit number of teachers
        n = task_label[-1]
        return f"evaluation/teacher{n}"
    elif task_label.startswith("export"):
        return "exported"
    elif task_label.startswith("train-teacher-model"):
        # the last character of these tasks is the teacher number
        # assumption: we will not have a double digit number of teachers
        n = task_label[-1]
        return f"teacher{n}"

    raise Exception(
        f"couldn't find `step_dir` for upstream task: {task_label}. upload_artifacts.py most likely needs to be updated."
    )


CHUNKED_TASK_PATTERN = re.compile(".*([0-9]+)/([0-9]+)$")


def get_chunk_number(task_label: str) -> str:
    m = CHUNKED_TASK_PATTERN.match(task_label)
    if not m:
        assert False, f"{task_label} is not a chunked task"
    return m.groups()[0]
