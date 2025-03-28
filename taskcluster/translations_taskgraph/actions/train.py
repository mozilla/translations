# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import json
import logging

from taskgraph.actions.registry import register_callback_action
from taskgraph.decision import taskgraph_decision
from taskgraph.parameters import Parameters
from taskgraph.taskgraph import TaskGraph
from taskgraph.util.taskcluster import get_ancestors, get_artifact

from translations_taskgraph.parameters import get_ci_training_config

logger = logging.getLogger(__name__)

TRAIN_ON_PROJECTS = (
    "https://github.com/mozilla/translations",
    "https://github.com/mozilla-releng/staging-firefox-translations-training",
)

WORKER_CLASSES = (
    # Regular, on-demand GCP instances
    "gcp-standard",
    # Spot instances in GCP
    "gcp-spot",
)


def can_train(parameters):
    return parameters["head_repository"] in TRAIN_ON_PROJECTS or (
        parameters["base_repository"] in TRAIN_ON_PROJECTS
        and parameters["tasks_for"].startswith("github-pull-request")
    )


defaults = get_ci_training_config()["training_config"]


def validate_model_continuation(params):
    pretrained_models = params["training_config"].get("continuation", {}).get("models", {})
    teacher = pretrained_models.get("teacher")
    if teacher:
        teacher_ensemble = params["training_config"]["experiment"]["teacher-ensemble"]
        if len(teacher["urls"]) != teacher_ensemble:
            raise Exception(
                f"The experiment's 'teacher-ensemble' ({teacher_ensemble}) "
                f"does not match the number of provided model 'urls' ({len(teacher['urls'])}) "
                f"for the pretrained 'train-teacher' ensemble."
            )


@register_callback_action(
    name="train",
    title="Train",
    symbol="train",
    description="Initiate part or all of the training pipeline",
    cb_name="train",
    permission="train",
    order=500,
    context=[],
    available=can_train,
    schema=lambda graph_config: {
        "type": "object",
        "properties": {
            "previous_group_ids": {
                "type": "array",
                "description": """Optional: an array of taskIds of decision or action
tasks from the previous group(s) to use to populate our `previous_group_kinds`.
Tasks specified here will be used as long as their label matches a needed task, and that
task is upstream of `start-stage`. (That is to say: even if a task from one of these groups
has a cache digest that doesn't match what the downstream task wants, it will still be used. This
can be used for quick iteration of functionality where the quality of the outputs is not important.)""",
                "items": {
                    "type": "string",
                },
            },
            "start-stage": {
                "type": "string",
                "description": """Optional: The stage of the pipeline to begin at, provided replacements
can be found for tasks upstream of this stage. Usually used in conjunction with `previous_group_ids`
which allows for specifying task group ids to fetch existing tasks from.""",
                "default": "",
                # We need to allow for no stage to be specified, in additional to all of the
                # valid stages.
                "enum": graph_config["valid-stages"] + [""],
            },
            "target-stage": {
                "type": "string",
                "description": """The stage of the pipeline to run until
(any stages this choice depends on will be automatically included).""",
                "default": defaults["target-stage"],
                "enum": graph_config["valid-stages"],
            },
            "wandb-publication": {
                "type": "boolean",
                "description": """Enable publication to Weights and Biases""",
                "default": True,
            },
            "experiment": {
                "type": "object",
                "default": defaults["experiment"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "A name for the experiment",
                    },
                    "src": {
                        "type": "string",
                        "description": "The src locale to train",
                    },
                    "trg": {
                        "type": "string",
                        "description": "The trg locale to train",
                    },
                    "teacher-ensemble": {
                        "type": "number",
                        "description": "Number of teachers to train",
                    },
                    "teacher-mode": {
                        "type": "string",
                        "description": "Teacher training mode",
                        "enum": ["one-stage", "two-stage"],
                        "default": "two-stage",
                    },
                    "teacher-decoder": {
                        "type": "string",
                        "description": "Translate with either Marian or CTranslate2",
                        "enum": ["marian", "ctranslate2"],
                        "default": "marian",
                    },
                    "student-model": {
                        "type": "string",
                        "description": "Student model configuration",
                        "enum": ["tiny", "base"],
                        "default": "tiny",
                    },
                    "mono-max-sentences-src": {
                        "type": "object",
                        "default": defaults["experiment"]["mono-max-sentences-src"],
                        "properties": {
                            "total": {
                                "type": "number",
                                "description": "limits for total src dataset",
                            },
                            "per-dataset": {
                                "type": "number",
                                "description": "limits per downloaded src dataset",
                            },
                        },
                    },
                    "mono-max-sentences-trg": {
                        "type": "object",
                        "default": defaults["experiment"]["mono-max-sentences-trg"],
                        "properties": {
                            "total": {
                                "type": "number",
                                "description": "limits for total trg dataset",
                            },
                            "per-dataset": {
                                "type": "number",
                                "description": "limits per downloaded trg dataset",
                            },
                        },
                    },
                    "spm-sample-size": {
                        "type": "number",
                        "description": "vocabularly training sample size",
                    },
                    "spm-vocab-size": {
                        "type": "number",
                        "description": "size of the vocabularly, can be reduced for testing",
                    },
                    "spm-vocab-split": {
                        "type": "boolean",
                        "description": "whether to separate SentencePiece vocabularies for source and target languages",
                    },
                    "best-model": {
                        "type": "string",
                        "description": "best model to use for training",
                    },
                    "use-opuscleaner": {
                        "type": "string",
                        "description": "use OpusCleaner to clean corpus",
                        "enum": ["true", "false"],
                    },
                    "opuscleaner-mode": {
                        "type": "string",
                        "description": "indicates whether to use dataset specific configs",
                        "enum": ["custom", "defaults"],
                        "default": "defaults",
                    },
                    "bicleaner": {
                        "properties": {
                            "default-threshold": {
                                "type": "number",
                                "description": "bicleaner threshold",
                            },
                            "dataset-thresholds": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "number",
                                },
                            },
                        },
                        "required": [
                            "default-threshold",
                        ],
                    },
                    "monocleaner": {
                        "properties": {
                            "mono-src": {
                                "type": "object",
                                "properties": {
                                    "default-threshold": {
                                        "type": "number",
                                        "description": "default monocleaner threshold for source language",
                                    },
                                    "dataset-thresholds": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "type": "number",
                                        },
                                    },
                                },
                                "required": [
                                    "default-threshold",
                                ],
                            },
                            "mono-trg": {
                                "type": "object",
                                "properties": {
                                    "default-threshold": {
                                        "type": "number",
                                        "description": "default monocleaner threshold for target language",
                                    },
                                    "dataset-thresholds": {
                                        "type": "object",
                                        "additionalProperties": {
                                            "type": "number",
                                        },
                                    },
                                },
                                "required": [
                                    "default-threshold",
                                ],
                            },
                        },
                        "required": [
                            "mono-src",
                            "mono-trg",
                        ],
                    },
                },
                "required": [
                    "name",
                    "src",
                    "trg",
                    "bicleaner",
                ],
            },
            "marian-args": {
                "type": "object",
                "default": defaults["marian-args"],
                "properties": {
                    "training-backward": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "string",
                        },
                    },
                    "training-teacher": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "string",
                        },
                    },
                    "training-student": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "string",
                        },
                    },
                    "training-student-finetuned": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "string",
                        },
                    },
                    "decoding-backward": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "string",
                        },
                    },
                    "decoding-teacher": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "string",
                        },
                    },
                },
            },
            "datasets": {
                "type": "object",
                "default": defaults["datasets"],
                "description": "The datasets to train with",
                "properties": {
                    "train": {
                        "type": "array",
                        "description": "Parallel training corpus",
                        "items": {
                            "type": "string",
                            # TODO
                            # "enum": []
                        },
                    },
                    "devtest": {
                        "type": "array",
                        "description": "datasets to merge for validation while training",
                        "items": {
                            "type": "string",
                            # TODO
                            # "enum": []
                        },
                    },
                    "test": {
                        "type": "array",
                        "description": "datasets for evaluation",
                        "items": {
                            "type": "string",
                            # TODO
                            # "enum": []
                        },
                    },
                    "mono-src": {
                        "type": "array",
                        "description": """
monolingual datasets (ex. paracrawl-mono_paracrawl8, commoncrawl_wmt16, news-crawl_news.2020)
to be translated by the teacher model
""",
                        "items": {
                            "type": "string",
                            # TODO
                            # "enum": []
                        },
                    },
                    "mono-trg": {
                        "type": "array",
                        "description": """
to be translated by the backward model to augment teacher corpus with back-translations
""",
                        "items": {
                            "type": "string",
                            # TODO
                            # "enum": []
                        },
                    },
                },
            },
            "continuation": {
                "type": "object",
                "default": {},
                "description": "Continue training from existing artifacts",
                "additionalProperties": False,
                "properties": {
                    "vocab": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "src": {"type": "string", "format": "uri"},
                            "trg": {"type": "string", "format": "uri"},
                        },
                        "required": ["src", "trg"],
                    },
                    "corpora": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "distillation": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "src": {"type": "string", "format": "uri"},
                                    "trg": {"type": "string", "format": "uri"},
                                    "tok-src": {"type": "string", "format": "uri"},
                                    "tok-trg": {"type": "string", "format": "uri"},
                                    "alignments": {"type": "string", "format": "uri"},
                                },
                                "required": ["src", "trg"],
                            },
                            "backtranslations": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "src": {"type": "string", "format": "uri"},
                                    "trg": {"type": "string", "format": "uri"},
                                    "tok-src": {"type": "string", "format": "uri"},
                                    "tok-trg": {"type": "string", "format": "uri"},
                                    "alignments": {"type": "string", "format": "uri"},
                                },
                                "required": ["src", "trg"],
                            },
                            "parallel": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "src": {"type": "string", "format": "uri"},
                                    "trg": {"type": "string", "format": "uri"},
                                    "tok-src": {"type": "string", "format": "uri"},
                                    "tok-trg": {"type": "string", "format": "uri"},
                                    "alignments": {"type": "string", "format": "uri"},
                                },
                                "required": ["src", "trg"],
                            },
                        },
                    },
                    # We are using urls because pretrained-models should be flexible enough
                    # to point at model (ensembles) that are not in taskcluster.
                    # Models could be in a long-term storage bucket, or we may use
                    # pretrained models hosted elsewhere.
                    "models": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "train-teacher": {
                                "type": "object",
                                "properties": {
                                    "urls": {
                                        "type": "array",
                                        "items": {"type": "string", "format": "uri"},
                                        "minItems": 1,
                                    },
                                    "mode": {
                                        "type": "string",
                                        "enum": ["continue", "init", "use"],
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": ["default", "opusmt"],
                                    },
                                },
                                "required": ["urls", "mode", "type"],
                            },
                            "train-backwards": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string"},
                                    "mode": {
                                        "type": "string",
                                        "enum": ["continue", "init", "use"],
                                    },
                                    "type": {
                                        "type": "string",
                                        "enum": ["default", "opusmt"],
                                    },
                                },
                                "required": ["url", "mode", "type"],
                            },
                        },
                    },
                },
            },
            "taskcluster": {
                "type": "object",
                "default": defaults["taskcluster"],
                "description": "Taskcluster-specific pipeline configuration, eg: chunking",
                "properties": {
                    "split-chunks": {
                        "type": "number",
                        "description": "The number of chunks (parallel jobs) to use in `split` steps",
                    },
                    "worker-classes": {
                        "type": "object",
                        "description": "The class of workers to use for this training, by kind",
                        "additionalProperties": {
                            "type": "string",
                            # TODO: add snakepit type(s) when they are brought online
                            "enum": ["gcp-standard", "gcp-spot"],
                        },
                    },
                },
            },
        },
        "required": [
            "target-stage",
            "datasets",
            "experiment",
            "marian-args",
        ],
    },
)
def train_action(parameters, graph_config, input, task_group_id, task_id):
    # TODO: Add a whack load of verification here. Things such as:
    # - datasets all exist
    # - locale pair exists for each dataset
    # - stage is valid
    # etc.

    parameters = dict(parameters)

    start_stage = input.pop("start-stage", None)
    if start_stage:
        if "previous_group_ids" not in input:
            raise Exception(
                "'previous_group_ids' is required to use 'start-stage' (otherwise we can't skip earlier tasks)"
            )

        previous_group_ids = input.pop("previous_group_ids")

        # First, we create one big graph out of all of the tasks from the specified group IDs.
        label_to_task_id = {}
        combined_full_task_graph = {}
        for graph_id in previous_group_ids:
            label_to_task_id.update(get_artifact(graph_id, "public/label-to-taskid.json"))
            full_task_graph = get_artifact(graph_id, "public/full-task-graph.json")
            combined_full_task_graph.update(full_task_graph)
        _, combined_full_task_graph = TaskGraph.from_json(combined_full_task_graph)

        # Next, we find the task id(s) corresponding of the tasks that match the stage
        # we want to start at.
        start_task_ids = []
        for label, task in combined_full_task_graph.tasks.items():
            if task.attributes.get("stage") == start_stage:
                start_task_ids.append(label_to_task_id[label])

        # Finally, we walk up the graph from our starting point and add any tasks found
        # as `existing_tasks`. These map task labels (eg: backtranslations-train-backwards-model-ru-en) to
        # task ids, and will be used instead of scheduling new tasks for any tasks with
        # an identical name.
        # As of taskgraph 13.0 `get_ancestors` returns taskids -> labels
        # `existing_tasks` needs the opposite
        parameters["existing_tasks"] = {v: k for k, v in get_ancestors(start_task_ids).items()}

    # Override the `existing_tasks` explicitly provided in the action's input
    existing_tasks = input.pop("existing_tasks", {})

    # Find and log `overridden_existing_tasks`
    overridden_existing_tasks = {
        existing_task: parameters["existing_tasks"][existing_task]
        for existing_task in existing_tasks.keys()
        if existing_task in parameters["existing_tasks"]
    }

    if overridden_existing_tasks:
        logger.info(
            f"Old values for `overridden_existing_tasks`: {json.dumps(overridden_existing_tasks, indent=2)}"
        )

    # Do the override!
    parameters["existing_tasks"].update(existing_tasks)

    # Log the new values for the `overridden_existing_tasks`
    new_values_for_overridden = {
        existing_task: parameters["existing_tasks"][existing_task]
        for existing_task in overridden_existing_tasks.keys()
    }

    if new_values_for_overridden:
        logger.info(
            f"New values for `overridden_existing_tasks`: {json.dumps(new_values_for_overridden, indent=2)}"
        )

    parameters["target_tasks_method"] = "train-target-tasks"
    parameters["optimize_target_tasks"] = True
    parameters["tasks_for"] = "action"
    parameters["training_config"] = input

    validate_model_continuation(parameters)

    parameters = Parameters(**parameters)
    taskgraph_decision({"root": graph_config.root_dir}, parameters=parameters)
