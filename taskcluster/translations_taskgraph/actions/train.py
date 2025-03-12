# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import json
import logging

from typing import Optional

from taskgraph.config import GraphConfig
from taskgraph.actions.registry import register_callback_action
from taskgraph.decision import taskgraph_decision
from taskgraph.parameters import Parameters
from taskgraph.taskgraph import TaskGraph
from taskgraph.util.taskcluster import get_ancestors, get_artifact

from translations_taskgraph.parameters import get_ci_training_config
from translations_taskgraph.training_config import (
    TrainingConfig,
    Parameters as ParametersDataClass,
)
from translations_taskgraph.util.dataclass_helpers import build_json_schema

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


def validate_pretrained_models(parameters: ParametersDataClass):
    """
    Validates that the pretrained models match the training config.
    """
    pretrained_models = parameters.training_config.experiment.pretrained_models
    train_teacher = pretrained_models.train_teacher
    if train_teacher:
        teacher_ensemble = parameters.training_config.experiment.teacher_ensemble
        if len(train_teacher.urls) != teacher_ensemble:
            raise Exception(
                f"The experiment's 'teacher-ensemble' ({teacher_ensemble}) "
                f"does not match the number of provided model 'urls' ({len(train_teacher.urls)}) "
                f"for the pretrained 'train-teacher' ensemble."
            )
    train_backwards = pretrained_models.train_backwards
    if train_backwards:
        if len(train_backwards.urls) != 1:
            raise Exception(
                f"The experiment's 'pretrained-models.backward.urls' ({len(train_backwards.urls)}) "
                f"must be equal to one (1). "
                f"The pipeline's backward model is _not_ an ensemble."
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
    schema=lambda graph_config: build_json_schema(TrainingConfig),
)
def train_action(
    parameters_dict: dict,
    graph_config: GraphConfig,
    training_config_dict: dict,
    task_group_id: Optional[str],
    task_id: Optional[str],
):
    """
    Consume a training config and kick off the train action.

    Arguments:
        parameters:
            The parameters for the action. Note that the training config is the CI config.
        graph_config:
            The Taskgraph GraphConfig class.
        training_config_dict:
            The training config to use for training.
        task_group_id:
            This will be not be set when running locally.
        task_id:
            This will be not be set when running locally.
    """
    parameters = ParametersDataClass.from_dict(parameters_dict)
    training_config = TrainingConfig.from_dict_validated(training_config_dict)

    start_stage = training_config.start_stage
    training_config.start_stage = None

    if start_stage:
        if training_config.previous_group_ids is None:
            raise Exception(
                "'previous-group-ids' is required to use 'start-stage' (otherwise we can't skip earlier tasks)"
            )

        previous_group_ids = training_config.previous_group_ids
        training_config.previous_group_ids = None

        # First, we create one big graph out of all of the tasks from the specified group IDs.
        label_to_task_id = {}
        combined_full_task_graph_dict = {}
        for graph_id in previous_group_ids:
            label_to_task_id.update(get_artifact(graph_id, "public/label-to-taskid.json"))
            full_task_graph = get_artifact(graph_id, "public/full-task-graph.json")
            combined_full_task_graph_dict.update(full_task_graph)

        _, combined_full_task_graph = TaskGraph.from_json(combined_full_task_graph_dict)

        # Next, we find the task id(s) corresponding of the tasks that match the stage
        # we want to start at.
        start_task_ids = []
        for label, task in combined_full_task_graph.tasks.items():
            if task.attributes.get("stage") == start_stage:
                start_task_ids.append(label_to_task_id[label])

        # Finally, we walk up the graph from our starting point and add any tasks found
        # as `existing_tasks`. These map task labels (eg: train-backwards-ru-en) to
        # task ids, and will be used instead of scheduling new tasks for any tasks with
        # an identical name.
        # As of taskgraph 13.0 `get_ancestors` returns taskids -> labels
        # `existing_tasks` needs the opposite
        parameters.existing_tasks = {v: k for k, v in get_ancestors(start_task_ids).items()}

    # Override the `existing_tasks` explicitly provided in the action's input
    existing_tasks = training_config.existing_tasks
    training_config.existing_tasks = {}

    # Find and log `overridden_existing_tasks`
    overridden_existing_tasks = {
        existing_task: parameters.existing_tasks[existing_task]
        for existing_task in existing_tasks.keys()
        if existing_task in parameters.existing_tasks
    }

    if overridden_existing_tasks:
        logger.info(
            f"Old values for `overridden_existing_tasks`: {json.dumps(overridden_existing_tasks, indent=2)}"
        )

    # Do the override!
    parameters.existing_tasks.update(existing_tasks)

    # Log the new values for the `overridden_existing_tasks`
    new_values_for_overridden = {
        existing_task: parameters.existing_tasks[existing_task]
        for existing_task in overridden_existing_tasks.keys()
    }

    if new_values_for_overridden:
        logger.info(
            f"New values for `overridden_existing_tasks`: {json.dumps(new_values_for_overridden, indent=2)}"
        )

    parameters.target_tasks_method = "train-target-tasks"
    parameters.optimize_target_tasks = True
    parameters.tasks_for = "action"

    # The training config is taskcluster/configs/config.ci.yml by default, replace it with
    # the train action's config.
    parameters.training_config = training_config

    validate_pretrained_models(parameters)
    parameters = Parameters(strict=True, repo_root=None, **parameters.to_dict())
    taskgraph_decision({"root": graph_config.root_dir}, parameters=parameters)
