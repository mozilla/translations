# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import logging
from pathlib import Path
from taskgraph.parameters import extend_parameters_schema
import yaml
import voluptuous
from translations_taskgraph.training_config import TrainingConfig
from translations_taskgraph.util.dataclass_helpers import build_voluptuous_schema


logger = logging.getLogger(__name__)


# By default, provide a very minimal config for CI that runs very quickly. This allows
# the pipeline to be validated in CI. The production training configs should override
# all of these values.
def get_ci_training_config(_=None) -> dict:
    vcs_path = (Path(__file__).parent / "../..").resolve()
    config_path = vcs_path / "taskcluster/configs/config.ci.yml"

    with config_path.open() as file:
        return {"training_config": yaml.safe_load(file)}


# Taskgraph expects a Voluptuous specification. These schemas aren't available to
# the Python type system. So instead use the type-safe TrainingConfig as the
# source of truth, and dynamically build the Voluptuous schema from the dataclass.
extend_parameters_schema(
    {voluptuous.Required("training_config"): build_voluptuous_schema(TrainingConfig)},
    defaults_fn=get_ci_training_config,
)


def deep_setdefault(dict_, defaults):
    for k, v in defaults.items():
        if isinstance(dict_.get(k), dict):
            deep_setdefault(dict_[k], defaults[k])
        else:
            dict_[k] = v


def get_decision_parameters(graph_config, parameters):
    parameters.setdefault("training_config", {})
    deep_setdefault(parameters, get_ci_training_config())
    # We run the pipeline on a cron schedule to enable integration testing when
    # worker images change (see https://bugzilla.mozilla.org/show_bug.cgi?id=1937882).
    # These runs should _never_ be sent to W&B to avoid cluttering it up
    # with data of no value.
    if (
        parameters["tasks_for"] == "cron"
        and parameters["target_tasks_method"] == "train-target-tasks"
    ):
        logger.info("Overriding wandb-publication to be False for cron pipeline run")
        parameters["training_config"]["wandb-publication"] = False
