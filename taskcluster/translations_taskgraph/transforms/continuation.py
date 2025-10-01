# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This transform is responsible for implementing corpus continuation, where previous
corpora from training runs can be re-used. This rewrites the dependencies for the
tasks in order to trim out tasks that are not needed. For instance providing all
the data to train a student, (vocab, corpora, backwards model) the teacher step
will be omitted from the training graph, and a vastly simplified graph will be
produced.
"""

from typing import Any, Iterable, Literal, Optional, TypedDict
from taskgraph.transforms.base import TransformSequence, TransformConfig

transforms = TransformSequence()


# This is minimally typed to just provide some hints for this transform.
Job = TypedDict(
    "Job",
    {
        "name": str,
        "fetches": Optional[dict[str, list[dict[str, Any]]]],
        "dependencies": dict[str, str],
        "from-deps": dict[str, Any],
        "task-context": dict[str, Any],
        "attributes": dict[str, Any],
    },
)


Vocab = TypedDict(
    "Vocab",
    {
        "src": str,
        "trg": str,
    },
)

BackwardsModel = TypedDict(
    "BackwardsModel",
    {
        "url": str,
        "mode": Literal["continue"] | Literal["init"] | Literal["use"],
        "type": Literal["default"] | Literal["opusmt"],
    },
)

Models = TypedDict(
    "Models",
    {
        "backwards": Optional[BackwardsModel],
    },
)

Corpus = TypedDict(
    "Corpus",
    {
        "src": str,
        "trg": str,
        "alignments": Optional[str],
        "tok-src": Optional[str],
        "tok-trg": Optional[str],
    },
)

Continuation = TypedDict(
    "Continuation",
    {
        "models": Optional[Models],
        "corpora": Optional[dict[str, Corpus]],
        "vocab": Optional[Vocab],
    },
)

transforms = TransformSequence()


def rewrite_dependencies(job: Job, old_task: str, new_task: str):
    # Rewrite the dependences
    # For example rewrite:
    #   dependencies:
    #       corpus-merge-parallel: corpus-merge-parallel-{src_locale}-{trg_locale}
    # To:
    #   dependencies:
    #       corpus-parallel: corpus-parallel-{src_locale}-{trg_locale}
    dependencies = job.get("dependencies", {})
    task_dependency = dependencies.pop(old_task, None)

    if task_dependency:
        dependencies[new_task] = new_task + "-{src_locale}-{trg_locale}"

    # Rewrite the fetches name to the new task.
    # For example here:
    #   fetches.corpus-merge-parallel -> fetches.corpus-parallel
    #
    # fetches:
    #     toolchain:
    #         - marian
    #     corpus-merge-parallel:
    #         - artifact: corpus.{src_locale}.zst
    #           extract: false
    #         - artifact: corpus.{trg_locale}.zst
    #           extract: false
    fetches = job.get("fetches") or {}
    artifacts = fetches.pop(old_task, None)
    if artifacts:
        fetches[new_task] = artifacts

    # Rewrite the fetches for "from-deps", which is the mechanism used for chunking.
    fetches = job.get("from-deps", {}).get("fetches", {})
    artifacts = fetches.pop(old_task, None)
    if artifacts:
        fetches[new_task] = artifacts

    # Replace any substitution fields that mention the old task.
    substitution_fields: list[str] = job.get("task-context", {}).get("substitution-fields", [])
    for key, value in enumerate(substitution_fields):
        substitution_fields[key] = value.replace(old_task, new_task)


@transforms.add
def apply_continuation(config: TransformConfig, jobs: Iterable[Job]):
    """
    When an existing corpus is available, rewriting the task graph to omit the steps
    needed to generate that corpus.

    Rewrites dependencies, e.g.:
        corpus-merge-parallel               -> continuation-corpus-parallel
        corpus-merge-mono-trg               -> continuation-corpus-backtranslations
        distillation-corpus-final-filtering -> continuation-corpus-distillation
    """

    # Uncomment these lines to allow for print debugging.
    # It's helpful to debug by running a test case in: tests/test_continuation.py

    # import sys
    # stdout = sys.stdout
    # sys.stdout = sys.__stdout__

    training_config: dict = config.params["training_config"]
    continuation: Continuation = training_config.get("continuation", {})

    corpora = continuation.get("corpora")
    corpus_backtranslations = validate_corpora_config(corpora, "backtranslations")
    corpus_parallel = validate_corpora_config(corpora, "parallel")
    corpus_distillation = validate_corpora_config(corpora, "distillation")

    vocab: Optional[Vocab] = continuation.get("vocab")
    models: Optional[Models] = continuation.get("models")
    # If the models are in the "use" mode and are the "default" type, they can be used
    # for changing dependencies of tasks.
    model_backwards: Optional[BackwardsModel] = None

    if models:
        model_backwards = models.get("backwards")

        if (
            not model_backwards
            or model_backwards["mode"] != "use"
            or model_backwards["type"] != "default"
        ):
            model_backwards = None

    for job in jobs:
        # The stage is often the identifier you want rather than job name, for instance,
        # "corpus-align-distillation" is the stage, while "{src_locale}-{trg_locale}" is the
        # actual job name at this point in time.
        stage = job.get("attributes", {}).get("stage", "")
        label = f"{stage}-{job['name']}"

        # Ensure continuation tasks don't get produced unless they are explicitly requested.
        if stage == "continuation-corpus":
            if (
                not corpus_backtranslations
                and label == "continuation-corpus-backtranslations-{src_locale}-{trg_locale}"
            ):
                continue
            if (
                not corpus_parallel
                and label == "continuation-corpus-parallel-{src_locale}-{trg_locale}"
            ):
                continue
            if (
                not corpus_distillation
                and label == "continuation-corpus-distillation-{src_locale}-{trg_locale}"
            ):
                continue
        if not vocab and stage == "continuation-vocab":
            continue
        if (
            not model_backwards
            and stage == "continuation-model"
            and label == "continuation-backwards-{src_locale}-{trg_locale}"
        ):
            continue

        if corpus_parallel:
            # Upload tasks are pulled in by `all-pipeline`, and in turn pull in the
            # pipeline tasks they depend on. Additionally, unlike the pipeline tasks
            # themselves, upload tasks are not linked to each other. Because of this
            # we must fully specify the upstream stages that should be skipped.
            if stage in {"corpus-merge-parallel", "corpus-clean-parallel-fetch-bicleaner-model"}:
                # Skip any jobs that should never be produced. This helps ensure
                # that if they do somehow get produced, the taskgraph will fail to
                # fully resolve.
                continue

            rewrite_dependencies(
                job,
                old_task="corpus-merge-parallel",
                new_task="continuation-corpus-parallel",
            )
            if corpus_parallel.get("alignments"):
                if stage in {
                    "corpus-align-parallel",
                    "backtranslations-mono-trg-chunk",
                    "backtranslations-mono-trg-dechunk-translations",
                    "backtranslations-mono-trg-translate",
                }:
                    continue
                rewrite_dependencies(
                    job,
                    old_task="corpus-align-parallel",
                    new_task="continuation-corpus-parallel",
                )

        if corpus_backtranslations:
            # Upload tasks are pulled in by `all-pipeline`, and in turn pull in the
            # pipeline tasks they depend on. Additionally, unlike the pipeline tasks
            # themselves, upload tasks are not linked to each other. Because of this
            # we must fully specify the upstream stages that should be skipped.
            if stage in {"corpus-merge-mono", "evaluate-backwards"}:
                # Skip any jobs that should never be produced. This helps ensure
                # that if they do somehow get produced, the taskgraph will fail to
                # fully resolve.
                continue

            rewrite_dependencies(
                job,
                old_task="corpus-merge-mono-trg",
                new_task="continuation-corpus-backtranslations",
            )
            rewrite_dependencies(
                job,
                old_task="corpus-merge-mono-src",
                new_task="continuation-corpus-backtranslations",
            )

            if corpus_backtranslations.get("alignments"):
                if stage in {
                    "corpus-align-backtranslations",
                    "backtranslations-mono-trg-chunk",
                    "backtranslations-mono-trg-dechunk-translations",
                    "backtranslations-mono-trg-translate",
                }:
                    continue
                rewrite_dependencies(
                    job,
                    old_task="corpus-align-backtranslations",
                    new_task="continuation-corpus-backtranslations",
                )

        if corpus_distillation:
            # Upload tasks are pulled in by `all-pipeline`, and in turn pull in the
            # pipeline tasks they depend on. Additionally, unlike the pipeline tasks
            # themselves, upload tasks are not linked to each other. Because of this
            # we must fully specify the upstream stages that should be skipped.
            if stage in {
                "backtranslations-mono-trg-chunk",
                "backtranslations-mono-trg-dechunk-translations",
                "backtranslations-mono-trg-translate",
                "backtranslations-train-backwards-model",
                "corpus-align-backtranslations",
                "corpus-align-parallel",
                "corpus-clean-mono",
                "corpus-clean-parallel",
                "corpus-clean-parallel-bicleaner-ai",
                "corpus-clean-parallel-fetch-bicleaner-model",
                "corpus-merge-distillation",
                "corpus-merge-mono",
                "corpus-merge-parallel",
                "distillation-corpus-final-filtering",
                "distillation-mono-src-chunk",
                "distillation-mono-src-dechunk-translations",
                "distillation-mono-src-translate",
                "distillation-parallel-src-chunk",
                "distillation-parallel-src-dechunk-translations",
                "distillation-parallel-src-extract-best",
                "distillation-parallel-src-translate",
                "distillation-parallel-src-translations-score",
                "distillation-src-translations-score",
                "evaluate-backwards",
                "evaluate-teacher",
                "evaluate-teacher-ensemble",
                "train-teacher-model",
            }:
                # Skip any jobs that should never be produced. This helps ensure
                # that if they do somehow get produced, the taskgraph will fail to
                # fully resolve.
                continue

            rewrite_dependencies(
                job,
                old_task="distillation-corpus-final-filtering",
                new_task="continuation-corpus-distillation",
            )
            if corpus_distillation.get("alignments"):
                if stage == "corpus-align-distillation":
                    continue
                rewrite_dependencies(
                    job,
                    old_task="corpus-align-distillation",
                    new_task="continuation-corpus-distillation",
                )

        if vocab:
            if stage == "build-vocab":
                # Skip any jobs that should never be produced. This helps ensure
                # that if they do somehow get produced, the taskgraph will fail to
                # fully resolve.
                continue

            rewrite_dependencies(job, old_task="build-vocab", new_task="continuation-vocab")

        if model_backwards:
            if stage in {"backtranslations-train-backwards-model", "evaluate-backwards"}:
                # Skip any jobs that should never be produced. This helps ensure
                # that if they do somehow get produced, the taskgraph will fail to
                # fully resolve.
                continue
            rewrite_dependencies(
                job,
                old_task="backtranslations-train-backwards-model",
                new_task="continuation-model-backwards",
            )

        # If alignments need to be re-generated, don't attempt to re-use alignment priors.
        if (corpus_distillation and not corpus_distillation.get("alignments")) or (
            corpus_backtranslations and not corpus_backtranslations.get("alignments")
        ):
            remove_alignment_priors_dependencies(job)

        yield job


def remove_alignment_priors_dependencies(job: Job):
    """
    Removes the following in case the corpus.priors are not available.

        dependencies:
            corpus-align-parallel: corpus-align-parallel-{src_locale}-{trg_locale}
        fetches:
            corpus-align-parallel:
                - artifact: corpus.priors
    """
    fetches = job.get("fetches")
    dependencies = job.get("dependencies")
    if not dependencies or not fetches:
        return
    alignments = fetches.get("corpus-align-parallel", [])
    if len(alignments) == 1 and alignments[0].get("artifact") == "corpus.priors":
        dependencies.pop("corpus-align-parallel")
        fetches.pop("corpus-align-parallel")


def validate_corpora_config(
    corpora: Optional[dict[str, Corpus]], corpus_key: str
) -> Optional[Corpus]:
    """
    Ensure that all of the files are defined if using an existing corpus.
    """
    if not corpora:
        return None

    corpus_files = corpora.get(corpus_key)

    if not corpus_files:
        return None

    def raise_error(file_key: str):
        raise ValueError(f'The "{file_key}" key was not found in the "corpora.{corpus_key}"')

    if "src" not in corpus_files:
        raise_error("src")
    if "trg" not in corpus_files:
        raise_error("trg")

    if "tok-src" in corpus_files or "tok-trg" in corpus_files or "alignments" in corpus_files:
        if "tok-src" not in corpus_files:
            raise_error("tok-src")
        if "tok-trg" not in corpus_files:
            raise_error("tok-src")
        if "alignments" not in corpus_files:
            raise_error("alignments")

    return corpus_files  # type: ignore[reportReturnType]
