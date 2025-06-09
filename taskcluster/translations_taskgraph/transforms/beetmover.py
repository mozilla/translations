# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# This transform has a very simple job: cast fields in a task definition from
# one type to another. The only reason it exists is because we have some fields
# that `task_context` fills in as a string, but that other transforms or code
# requires to be an int.

from taskgraph.transforms.base import TransformSequence
from taskgraph.util.schema import resolve_keyed_by

from translations_taskgraph.util.dependencies import get_beetmover_upstream_dependency
from translations_taskgraph.util.substitution import substitute

transforms = TransformSequence()


@transforms.add
def evaluate_keyed_by(config, jobs):
    for job in jobs:
        upstream_kind = list(job["dependencies"].keys())[0]
        for field in ("upstream-artifacts", "bucket", "artifact-map"):
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
def substitute_step_dir(config, jobs):
    for job in jobs:
        upstream_task = get_beetmover_upstream_dependency(job["dependencies"])[1]
        # massage the upstream task name to be the generic form that we have in our mapping
        src = config.params["training_config"]["experiment"]["src"]
        trg = config.params["training_config"]["experiment"]["trg"]
        pair = f"-{src}-{trg}"
        if pair in upstream_task:
            upstream_task = upstream_task.replace(pair, "-SRC-TRG")
        elif f"-{src}" in upstream_task:
            upstream_task = upstream_task.replace(f"-{src}", "-SRC")
        elif f"-{trg}" in upstream_task:
            upstream_task = upstream_task.replace(f"-{trg}", "-TRG")

        if upstream_task in TASK_TO_STEP_DIR:
            step_dir = TASK_TO_STEP_DIR[upstream_task]
        else:
            # some tasks don't have a one to one mapping; they need to be handled
            # specially
            if upstream_task.startswith("backtranslations-mono-trg-translate"):
                # full task is like: backtranslations-mono-trg-translate-SRC-TRG-1/2
                # non-log artifacts could share the same directory; logs would overwrite
                # each other though.
                step_dir = "???"
            elif upstream_task.startswith("corpus-analyze-mono"):
                step_dir = "???"
            elif upstream_task.startswith("corpus-analyze-parallel"):
                step_dir = "???"
            elif upstream_task.startswith("corpus-clean-mono"):
                step_dir = "???"
            elif upstream_task.startswith("corpus-clean-parallel-bicleaner"):
                step_dir = "???"
            elif upstream_task.startswith("corpus-clean-parallel"):
                step_dir = "???"
            elif upstream_task.startswith("distillation-mono-src-translate"):
                step_dir = "???"
            elif upstream_task.startswith("distillation-parallel-src-extract-best"):
                step_dir = "???"
            elif upstream_task.startswith("distillation-parallel-src-translate"):
                step_dir = "???"
            # evaluation tasks have many names, based on the dataset being used,
            # but they all publish to the same directories (with distinct filenames)
            elif upstream_task.startswith("evaluate-teacher"):
                # assumption: we will not have a double digit number of teachers
                n = upstream_task[-1]
                step_dir = f"evaluation/teacher{n}"
            elif upstream_task.startswith("evaluate-backward"):
                step_dir = "evaluation/backward"
            elif upstream_task.startswith("evaluate-finetuned-student"):
                step_dir = "evaluation/student-finetuned"
            elif upstream_task.startswith("evaluate-quantized"):
                step_dir = "evaluation/quantized"
            elif upstream_task.startswith("evaluate-student"):
                step_dir = "evaluation/student"
            elif upstream_task.startswith("evaluate-teacher-ensemble"):
                step_dir = "evaluation/teacher-ensemble"
            elif upstream_task.startswith("train-teacher-model"):
                # assumption: we will not have a double digit number of teachers
                n = upstream_task[-1]
                step_dir = f"teacher{n}"
            else:
                raise Exception(
                    f"couldn't find `step_dir` for upstream task: {upstream_task}. beetmover.py most likely needs to be updated."
                )

        substitute(job["worker"]["artifact-map"], step_dir=step_dir)
        yield job


TASK_TO_STEP_DIR = {
    "backtranslations-mono-trg-chunk-TRG": "???",
    "backtranslations-mono-trg-dechunk-translations-SRC-TRG": "???",
    "backtranslations-train-backwards-model-SRC-TRG": "backward",
    "build-vocab-SRC-TRG": "vocab",
    "corpus-align-backtranslations-SRC-TRG": "???",
    "corpus-align-parallel-SRC-TRG": "???",
    "corpus-clean-parallel-fetch-bicleaner-model-SRC-TRG": "???",
    "corpus-merge-devset-SRC-TRG": "???",
    "corpus-merge-distillation-SRC-TRG": "???",
    "corpus-merge-mono-src-SRC": "???",
    "corpus-merge-mono-trg-TRG": "???",
    "corpus-merge-parallel-SRC-TRG": "???",
    "distillation-corpus-build-shortlist-SRC-TRG": "???",
    "distillation-corpus-final-filtering-SRC-TRG": "???",
    "distillation-mono-src-chunk-SRC": "???",
    "distillation-mono-src-dechunk-translations-SRC-TRG": "???",
    "distillation-parallel-src-chunk-SRC-TRG": "???",
    "distillation-parallel-src-dechunk-translations-SRC-TRG": "???",
    "distillation-parallel-src-translations-score-SRC-TRG": "???",
    "distillation-student-model-finetune-SRC-TRG": "student-finetuned",
    "distillation-student-model-quantize-SRC-TRG": "quantized",
    "distillation-student-model-train-SRC-TRG": "student",
    "export-SRC-TRG": "exported",
}
