# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# This transform has a very simple job: cast fields in a task definition from
# one type to another. The only reason it exists is because we have some fields
# that `task_context` fills in as a string, but that other transforms or code
# requires to be an int.

from taskgraph.transforms.base import TransformSequence
from taskgraph.util.schema import Schema, resolve_keyed_by
from voluptuous import ALLOW_EXTRA, Optional

from translations_taskgraph.util.substitution import substitute

transforms = TransformSequence()

@transforms.add
def verify_dependencies(config, jobs):
    for job in jobs:
        if len(job["dependencies"]) != 1:
            raise Exception("beetmover tasks must have exactly one dependency")

        yield job

@transforms.add
def add_task_id(config, jobs):
    for job in jobs:
        upstream_kind = list(job["dependencies"].keys())[0]
        job["worker"]["upstream-artifacts"][0]["taskId"] = {"task-reference": f"<{upstream_kind}>"}
        job["worker"]["artifact-map"][0]["taskId"] = {"task-reference": f"<{upstream_kind}>"}

        yield job

@transforms.add
def substitute_step_dir(config, jobs):
    for job in jobs:
        # TODO: make this correct for all stages
        substitute(job["worker"]["artifact-map"], step_dir=list(job["dependencies"].values())[0])
        yield job

@transforms.add
def evaluate_keyed_by(config, jobs):
    for job in jobs:
        upstream_kind = list(job["dependencies"].keys())[0]
        resolve_keyed_by(
            job["worker"]["upstream-artifacts"][0],
            "paths",
            item_name=job["description"],
            **{
                "upstream-kind": upstream_kind,
            }
        )

        resolve_keyed_by(
            job["worker"],
            "dryrun",
            item_name=job["description"],
            **{
                "tasks-for": config.params["tasks_for"],
            }
        )

        yield job
