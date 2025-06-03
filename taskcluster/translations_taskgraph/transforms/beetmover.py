# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# This transform is responsible for:
# * Evaluating the `keyed-by` expressions in the `beetmover` kind
# * Substituting in the `step_dir` variable in the `artifact-map`

from taskgraph.transforms.base import TransformSequence
from taskgraph.util.schema import resolve_keyed_by

from translations_taskgraph.util.dependencies import get_beetmover_upstream_dependency
from translations_taskgraph.util.substitution import substitute

transforms = TransformSequence()


@transforms.add
def evaluate_keyed_by(_, jobs):
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
def substitute_step_dir(_, jobs):
    for job in jobs:
        upstream_task = get_beetmover_upstream_dependency(job["dependencies"])[1]
        step_dir = task_to_step_dir(upstream_task)
        substitute(job["worker"]["artifact-map"], step_dir=step_dir)
        yield job


def task_to_step_dir(task_label: str) -> str:
    if task_label.startswith("backtranslations-mono-trg-chunk"):
        return "backtranslation-mono-trg"
    elif task_label.startswith("backtranslations-mono-trg-dechunk-translations"):
        return "backtranslations-mono-trg-dechunk"
    elif task_label.startswith("backtranslations-mono-trg-translate"):
        n = task_label[-3]
        return f"backtranslations-mono-trg-translate{n}"
    elif task_label.startswith("backtranslations-train-backwards-model"):
        return "backward"
    elif task_label.startswith("build-vocab"):
        return "vocab"
    elif task_label.startswith("corpus-align-backtranslations"):
        return "corpus-align-backtranslations"
    elif task_label.startswith("corpus-align-parallel"):
        return "corpus-align-parallel"
    elif task_label.startswith("corpus-analyze-mono"):
        # full task label is needed to avoid overwriting logs from different
        # datasets
        return task_label
    elif task_label.startswith("corpus-analyze-parallel"):
        # full task label is needed to avoid overwriting logs from different
        # datasets
        return task_label
    elif task_label.startswith("corpus-clean-mono"):
        # full task label is needed to avoid overwriting logs from different
        # datasets
        return task_label
    elif task_label.startswith("corpus-clean-parallel-bicleaner-ai"):
        # full task label is needed to avoid overwriting logs from different
        # datasets
        return task_label
    elif task_label.startswith("corpus-clean-parallel-fetch-bicleaner-model"):
        return "corpus-clean-parallel-fetch-bicleaner-model"
    elif task_label.startswith("corpus-clean-parallel"):
        # full task label is needed to avoid overwriting logs from different
        # datasets
        return task_label
    elif task_label.startswith("corpus-merge-devset"):
        return "corpus-merge-devset"
    elif task_label.startswith("corpus-merge-distillation"):
        return "corpus-merge-distillation"
    elif task_label.startswith("corpus-merge-mono-src"):
        return "corpus-merge-mono-src"
    elif task_label.startswith("corpus-merge-mono-trg"):
        return "corpus-merge-mono-trg"
    elif task_label.startswith("corpus-merge-parallel"):
        return "corpus-merge-parallel"
    elif task_label.startswith("distillation-corpus-build-shortlist"):
        return "distillation-corpus-build-shortlist"
    elif task_label.startswith("distillation-corpus-final-filtering"):
        return "distillation-corpus-final-filtering"
    elif task_label.startswith("distillation-mono-src-chunk"):
        return "distillation-mono-src-chunk"
    elif task_label.startswith("distillation-mono-src-dechunk-translations"):
        return "distillation-mono-src-dechunk"
    elif task_label.startswith("distillation-mono-src-translate"):
        n = task_label[-3]
        return f"distillation-mono-src-translate{n}"
    elif task_label.startswith("distillation-parallel-src-chunk"):
        return "distillation-parallel-src-chunk"
    elif task_label.startswith("distillation-parallel-src-dechunk-translations"):
        return "distillation-parallel-src-dechunk"
    elif task_label.startswith("distillation-parallel-src-extract-best"):
        n = task_label[-3]
        return f"distillation-parallel-src-extract-best{n}"
    elif task_label.startswith("distillation-parallel-src-translate"):
        n = task_label[-3]
        return f"distillation-parallel-src-translate{n}"
    elif task_label.startswith("distillation-parallel-src-translations-score"):
        return "distillation-parallel-src-translations-score"
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
        # assumption: we will not have a double digit number of teachers
        n = task_label[-1]
        return f"evaluation/teacher{n}"
    elif task_label.startswith("export"):
        return "exported"
    elif task_label.startswith("train-teacher-model"):
        # assumption: we will not have a double digit number of teachers
        n = task_label[-1]
        return f"teacher{n}"

    raise Exception(
        f"couldn't find `step_dir` for upstream task: {task_label}. beetmover.py most likely needs to be updated."
    )
