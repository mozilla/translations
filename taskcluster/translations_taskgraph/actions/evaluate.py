# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from typing import Any
from taskgraph.actions.registry import register_callback_action
from translations_taskgraph.actions.train import can_train
from taskgraph.config import GraphConfig
from taskgraph.parameters import Parameters


def get_evaluate_schema(graph_config: dict[str, Any]):
    return {
        "type": "object",
        "properties": {},
        "required": [],
    }


@register_callback_action(
    name="evaluate",
    title="Evaluate",
    symbol="evaluate",
    description="Run final evaluation.",
    cb_name="evaluate",
    permission="train",
    order=501,
    context=[],
    available=can_train,
    schema=get_evaluate_schema,
)
def train_action(
    parameters: Parameters,
    graph_config: GraphConfig,
    input: dict[str, Any],
    _,
    __,
) -> None:
    from taskgraph.decision import taskgraph_decision

    parameters_dict: dict[str, Any] = dict(parameters)

    # the target tasks method to be used by `taskgraph_decision`.
    # this function is registered in `target_tasks.py`, and is used
    # to select which generated tasks should actually be created
    parameters_dict["target_tasks_method"] = "final-eval-target-tasks"
    parameters_dict["tasks_for"] = "action"
    # pull out data from the action input. this comes from the payload when
    # firing an action. This data is made available to transforms and the
    # target tasks method through `config.params` and `parameters` respectively.
    parameters_dict["eval_config"] = input
    # used by the target tasks method to find desired tasks; there is no
    # connection between this target stage and the one in the `train` action
    # other than the naming convention
    parameters_dict["target-stage"] = "final-eval"

    # run the decision task. this generates tasks, selects target tasks, and
    # eventually schedules them to run.
    taskgraph_decision({"root": graph_config.root_dir}, parameters=Parameters(**parameters_dict))
