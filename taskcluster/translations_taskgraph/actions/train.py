# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import json
import logging
import os

from taskgraph.actions.registry import register_callback_action
from taskgraph.config import GraphConfig
from taskgraph.decision import taskgraph_decision
from taskgraph.parameters import Parameters
from taskgraph.util.taskcluster import get_ancestors
from typing import Any, cast, Mapping, Sequence

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


def can_train(parameters: dict[str, Any]) -> bool:
    return parameters["head_repository"] in TRAIN_ON_PROJECTS or (
        parameters["base_repository"] in TRAIN_ON_PROJECTS
        and parameters["tasks_for"].startswith("github-pull-request")
    )


def validate_continuation(params: dict[str, Any]) -> None:
    continuation = pretrained_models = params["training_config"].get("continuation", {})
    pretrained_models = continuation.get("models", {})
    corpora = continuation.get("corpora", {})
    datasets: dict[str, Any] = params["training_config"].get("datasets", {})

    teacher = pretrained_models.get("teacher")
    if teacher:
        teacher_ensemble = params["training_config"]["experiment"]["teacher-ensemble"]
        if len(teacher["urls"]) != teacher_ensemble:
            raise ValueError(
                f"The experiment's 'teacher-ensemble' ({teacher_ensemble}) "
                f"does not match the number of provided model 'urls' ({len(teacher['urls'])}) "
                f"for the pretrained 'train-teacher' ensemble."
            )

    distillation = corpora.get("distillation")
    if distillation:
        if "train" in datasets or "mono-src" in datasets or "mono-trg" in datasets:
            raise ValueError(
                'The "train", "mono-src", and "mono-trg" datasets should not be present '
                'in a training config that has "continuation.corpus.distillation".'
            )

        if "vocab" not in continuation:
            raise ValueError(
                'A vocab must be provided when using "continuation.corpus.distillation".'
            )


def get_descendants(task_id, queue, res=None):
    """
    Traverse the graph of dependent tasks and return a mapping
    from each descendant task ID to its name.
    """
    # only initialize on the very first call
    if res is None:
        res = {}

    resp = queue.listDependentTasks(task_id)

    # for each direct dependent, add it if unseen and recurse
    for t in resp.get("tasks", []):
        dep_id = t["status"]["taskId"]
        dep_name = t["task"]["metadata"]["name"]

        # skip anything we've already pulled in
        if dep_id in res:
            continue

        res[dep_id] = dep_name
        # recurse down to this child’s dependents
        get_descendants(dep_id, queue, res)

    return res


def get_train_parameters(
    parameters_obj: Parameters, training_config: dict[str, Any]
) -> Parameters:
    parameters: dict[str, Any] = dict(parameters_obj)

    previous_group_ids = training_config.pop("previous-group-ids", None)
    if previous_group_ids:
        # Resume the pipeline by reusing the completed tasks from previous task groups
        import taskcluster

        queue = taskcluster.Queue(options={"rootUrl": os.environ["TASKCLUSTER_ROOT_URL"]})
        tasks_to_add = {}
        for group_id in previous_group_ids:
            group = queue.listTaskGroup(group_id)

            if group is None:
                raise ValueError(f"Task group with ID {group_id} is not found")

            for task in cast(Sequence[Mapping[str, Any]], group["tasks"]):
                if task["status"]["state"] != "completed":
                    continue

                task_id = task["status"]["taskId"]
                task_name = task["task"]["metadata"]["name"]
                # Skip service tasks
                if (
                    task_name.startswith("Action")
                    or task_name.startswith("Decision")
                    or task_name.startswith("PR")
                ):
                    continue

                # Tasks from the latter previous-group-ids groups override previously found tasks
                tasks_to_add[task_name] = task_id

        logger.info(f"Found top level existing tasks`: {json.dumps(tasks_to_add, indent=2)}")

        if tasks_to_add:
            # Add ancestors of all the top level completed tasks
            for task_id, label in get_ancestors(list(tasks_to_add.values())).items():
                # Skip service tasks
                if (
                    label.startswith("Action")
                    or label.startswith("Decision")
                    or label.startswith("PR")
                ):
                    continue
                tasks_to_add[label] = task_id

            logger.info(f"Added ancestor tasks`: {json.dumps(tasks_to_add, indent=2)}")

            # Optionally rerun the tasks with the specified prefix and their descendants by removing them from the existing tasks
            start_prefix = training_config.pop("start-task-prefix", None)
            if start_prefix:
                start_tasks = {
                    id: label
                    for label, id in tasks_to_add.items()
                    if label.startswith(start_prefix)
                }
                logger.info(
                    f"Identified start tasks to rerun`: {json.dumps(start_tasks, indent=2)}"
                )
                all_descendants = {}
                for task_id in start_tasks:
                    descendants = get_descendants(task_id, queue)
                    all_descendants.update(descendants)

                logger.info(
                    f"Found start task descendants to remove`: {json.dumps(all_descendants, indent=2)}"
                )

                tasks_to_add = {
                    label: task_id
                    for label, task_id in tasks_to_add.items()
                    if task_id not in start_tasks and task_id not in all_descendants
                }
            parameters["existing_tasks"] = tasks_to_add

    # Override the `existing_tasks` explicitly provided in the action's input
    existing_tasks: dict[str, str] = training_config.pop("existing_tasks", {})

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
    parameters["existing_tasks"].update(existing_tasks)  # type: ignore
    logger.info(f'Final existing tasks: {json.dumps(parameters["existing_tasks"], indent=2)}')

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
    parameters["training_config"] = training_config

    validate_continuation(parameters)

    return Parameters(**parameters)


def make_schema_strict(schema: dict) -> dict:
    """
    This makes all keys required unless they are explicitly marked as optional, and
    it makes it so that no additional properties are allowed unless it's explicitly
    marked as being OK.
    """
    if schema.get("type") == "object" and "properties" in schema:
        if "required" in schema:
            raise Exception('Use {"optional": True} rather than the required array')

        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False

        properties: dict[str, Any] = schema["properties"]
        schema["required"] = [
            key
            for key, value in properties.items()
            if isinstance(value, dict) and not value.get("optional", False)
        ]

        # The "optional" property is custom to this funct
        schema.pop("optional", None)

        for prop in schema["properties"].values():
            if isinstance(prop, dict):
                make_schema_strict(prop)

    elif schema.get("type") == "array" and "items" in schema:
        # Recurse into array items if they're object schemas
        items = schema["items"]
        if isinstance(items, dict):
            make_schema_strict(items)
        elif isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    make_schema_strict(item)

    return schema


def get_training_config_schema(graph_config: dict[str, Any]):
    """
    The schema for the training config. The graph_config parameter is the
    taskcluster/config.yml file. For documentation of the elements see the
    `taskcluster/configs/config.prod.yml` which is the reference config.
    """
    schema = {
        "type": "object",
        "properties": {
            "previous-group-ids": {
                "type": "array",
                "items": {"type": "string"},
                "optional": True,  # respected by `make_schema_strict`
            },
            "start-task-prefix": {
                "type": "string",
                # We need to allow for no stage to be specified, in additional to all of the
                # valid stages.
                "enum": graph_config["valid-stages"] + [""],
                "optional": True,  # respected by `make_schema_strict`
            },
            "target-stage": {
                "type": "string",
                "enum": graph_config["valid-stages"],
            },
            "wandb-publication": {"type": "boolean"},
            "experiment": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "src": {"type": "string"},
                    "trg": {"type": "string"},
                    "archive-corpora": {"type": "boolean"},
                    "teacher-ensemble": {"type": "number"},
                    "teacher-mode": {
                        "type": "string",
                        "enum": ["one-stage", "two-stage"],
                    },
                    "teacher-decoder": {
                        "type": "string",
                        "enum": ["marian", "ctranslate2"],
                    },
                    "corpus-max-sentences": {
                        "type": "number",
                        "optional": True,
                    },
                    "student-model": {
                        "type": "string",
                        "enum": ["tiny", "base", "base-memory"],
                    },
                    "mono-max-sentences-src": {
                        "type": "object",
                        "properties": {
                            "total": {"type": "number"},
                            "per-dataset": {"type": "number"},
                        },
                    },
                    "mono-max-sentences-trg": {
                        "type": "object",
                        "properties": {
                            "total": {"type": "number"},
                            "per-dataset": {"type": "number"},
                        },
                    },
                    "spm-sample-size": {"type": "number"},
                    "spm-vocab-size": {"type": "number"},
                    "spm-vocab-split": {"type": "boolean"},
                    "best-model": {"type": "string"},
                    "opuscleaner-mode": {
                        "type": "string",
                        "enum": ["custom", "defaults"],
                    },
                    "bicleaner": {
                        "properties": {
                            "default-threshold": {"type": "number"},
                            "dataset-thresholds": {
                                "type": "object",
                                "additionalProperties": {"type": "number"},
                            },
                        },
                    },
                    "hplt-min-doc-score": {
                        "type": "object",
                        "properties": {
                            "mono-src": {"type": "number"},
                            "mono-trg": {"type": "number"},
                        },
                    },
                    "monocleaner": {
                        "properties": {
                            "mono-src": {
                                "type": "object",
                                "properties": {
                                    "default-threshold": {"type": "number"},
                                    "dataset-thresholds": {
                                        "type": "object",
                                        "additionalProperties": {"type": "number"},
                                    },
                                },
                            },
                            "mono-trg": {
                                "type": "object",
                                "properties": {
                                    "default-threshold": {"type": "number"},
                                    "dataset-thresholds": {
                                        "type": "object",
                                        "additionalProperties": {"type": "number"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            "marian-args": {
                "type": "object",
                "properties": {
                    "training-backward": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "training-teacher": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "training-student": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "training-student-finetuned": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "decoding-backward": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "decoding-teacher": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
            },
            "datasets": {
                "type": "object",
                "properties": {
                    "train": {
                        "type": "array",
                        "items": {"type": "string"},
                        "optional": True,  # Optional for training continuation
                    },
                    "devtest": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "test": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "mono-src": {
                        "type": "array",
                        "items": {"type": "string"},
                        "optional": True,  # Optional for training continuation
                    },
                    "mono-trg": {
                        "type": "array",
                        "items": {"type": "string"},
                        "optional": True,  # Optional for training continuation
                    },
                },
            },
            "continuation": {
                "type": "object",
                "optional": True,
                "properties": {
                    "vocab": {
                        "type": "object",
                        "properties": {
                            "src": {"type": "string", "format": "uri"},
                            "trg": {"type": "string", "format": "uri"},
                        },
                        "optional": True,
                    },
                    "corpora": {
                        "type": "object",
                        "optional": True,
                        "properties": {
                            "distillation": {
                                "type": "object",
                                "optional": True,
                                "properties": {
                                    "src": {"type": "string", "format": "uri"},
                                    "trg": {"type": "string", "format": "uri"},
                                    "tok-src": {
                                        "type": "string",
                                        "format": "uri",
                                        "optional": True,
                                    },
                                    "tok-trg": {
                                        "type": "string",
                                        "format": "uri",
                                        "optional": True,
                                    },
                                    "alignments": {
                                        "type": "string",
                                        "format": "uri",
                                        "optional": True,
                                    },
                                },
                            },
                            "backtranslations": {
                                "type": "object",
                                "optional": True,
                                "properties": {
                                    "src": {"type": "string", "format": "uri"},
                                    "trg": {"type": "string", "format": "uri"},
                                    "tok-src": {
                                        "type": "string",
                                        "format": "uri",
                                        "optional": True,
                                    },
                                    "tok-trg": {
                                        "type": "string",
                                        "format": "uri",
                                        "optional": True,
                                    },
                                    "alignments": {
                                        "type": "string",
                                        "format": "uri",
                                        "optional": True,
                                    },
                                },
                            },
                            "parallel": {
                                "type": "object",
                                "optional": True,
                                "properties": {
                                    "src": {"type": "string", "format": "uri"},
                                    "trg": {"type": "string", "format": "uri"},
                                    "tok-src": {
                                        "type": "string",
                                        "format": "uri",
                                        "optional": True,
                                    },
                                    "tok-trg": {
                                        "type": "string",
                                        "format": "uri",
                                        "optional": True,
                                    },
                                    "alignments": {
                                        "type": "string",
                                        "format": "uri",
                                        "optional": True,
                                    },
                                },
                            },
                        },
                    },
                    # We are using urls because pretrained-models should be flexible enough
                    # to point at model (ensembles) that are not in taskcluster.
                    # Models could be in a long-term storage bucket, or we may use
                    # pretrained models hosted elsewhere.
                    "models": {
                        "type": "object",
                        "optional": True,
                        "properties": {
                            "teacher": {
                                "type": "object",
                                "optional": True,
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
                            },
                            "backwards": {
                                "type": "object",
                                "optional": True,
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
                            },
                        },
                    },
                },
            },
            "taskcluster": {
                "type": "object",
                "properties": {
                    "split-chunks": {"type": "number"},
                    "upload-bucket": {
                        "type": "string",
                        "enum": ["development", "production"],
                    },
                    "worker-classes": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "string",
                            "enum": ["gcp-standard", "gcp-spot"],
                        },
                    },
                },
            },
        },
    }

    make_schema_strict(schema)

    return schema


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
    schema=get_training_config_schema,
)
def train_action(
    parameters: Parameters,
    graph_config: GraphConfig,
    input: dict[str, Any],
    task_group_id: str,
    task_id: str,
) -> None:
    """
    Generate the "train" action which kicks off training. The input parameters is
    the training config.

    https://taskcluster-taskgraph.readthedocs.io/en/latest/howto/create-actions.html#defining-action-tasks
    """
    taskgraph_decision(
        {"root": graph_config.root_dir, "verbose": True}, parameters=get_train_parameters(parameters, input)
    )
