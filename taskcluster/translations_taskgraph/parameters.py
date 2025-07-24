# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import logging
from pathlib import Path
from taskgraph.parameters import extend_parameters_schema
from voluptuous import Extra, Optional, Required
import yaml

logger = logging.getLogger(__name__)


# By default, provide a very minimal config for CI that runs very quickly. This allows
# the pipeline to be validated in CI. The production training configs should override
# all of these values.
def get_ci_training_config(_=None) -> dict:
    vcs_path = (Path(__file__).parent / "../..").resolve()
    config_path = vcs_path / "taskcluster/configs/config.ci.yml"

    with config_path.open() as file:
        return {"training_config": yaml.safe_load(file)}


extend_parameters_schema(
    {
        Required("training_config"): {
            Required("target-stage"): str,
            Required("marian-args"): {
                Optional("training-backward"): {str: str},
                Optional("training-teacher"): {str: str},
                Optional("training-student"): {str: str},
                Optional("training-student-finetuned"): {str: str},
                Optional("decoding-backward"): {str: str},
                Optional("decoding-teacher"): {str: str},
            },
            Required("experiment"): {
                Required("name"): str,
                Required("src"): str,
                Required("trg"): str,
                Required("archive-corpora"): bool,
                Required("teacher-ensemble"): int,
                Required("teacher-mode"): str,
                Required("teacher-decoder"): str,
                Required("student-model"): str,
                Optional("corpus-max-sentences"): int,
                Required("mono-max-sentences-trg"): {
                    Required("total"): int,
                    Required("per-dataset"): int,
                },
                Required("mono-max-sentences-src"): {
                    Required("total"): int,
                    Required("per-dataset"): int,
                },
                Required("spm-sample-size"): int,
                Optional("spm-vocab-size"): int,
                Required("spm-vocab-split"): bool,
                Required("best-model"): str,
                Optional("opuscleaner-mode"): str,
                Required("bicleaner"): {
                    Required("default-threshold"): float,
                    Optional("dataset-thresholds"): {
                        str: float,
                    },
                },
                Required("monocleaner"): {
                    Required("mono-src"): {
                        Required("default-threshold"): float,
                        Optional("dataset-thresholds"): {
                            str: float,
                        },
                    },
                    Required("mono-trg"): {
                        Required("default-threshold"): float,
                        Optional("dataset-thresholds"): {
                            str: float,
                        },
                    },
                },
                Required("hplt-min-doc-score"): {
                    Required("mono-src"): float,
                    Required("mono-trg"): float,
                },
            },
            Optional("datasets"): {
                str: [str],
            },
            Optional("continuation"): {
                Optional("vocab"): {
                    Required("src"): str,
                    Required("trg"): str,
                },
                Optional("models"): {
                    Optional("teacher"): {
                        Required("urls"): [str],
                        Required("mode"): str,
                        Required("type"): str,
                    },
                    Optional("backwards"): {
                        Required("url"): str,
                        Required("mode"): str,
                        Required("type"): str,
                    },
                },
                Optional("corpora"): {
                    Optional("backtranslations"): {
                        Required("src"): str,
                        Required("trg"): str,
                        Optional("aln"): str,
                    },
                    Optional("parallel"): {
                        Required("src"): str,
                        Required("trg"): str,
                        Optional("aln"): str,
                    },
                    Optional("distillation"): {
                        Required("src"): str,
                        Required("trg"): str,
                        Optional("aln"): str,
                    },
                },
            },
            Optional("taskcluster"): {
                Optional("split-chunks"): int,
                Optional("upload-bucket"): str,
                Required("worker-classes"): {
                    Required("default"): str,
                    Extra: str,
                },
            },
            Optional("wandb-publication"): bool,
        },
    },
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
