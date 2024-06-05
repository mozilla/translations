# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from taskgraph.transforms.task import payload_builder


@payload_builder(
    "beetmover-translations",
    schema={
        # TODO: flesh out schema
        "dryrun": bool,
        "release-properties": dict,
        "upstream-artifacts": list,
        "artifact-map": list,
    },
)
def build_beetmover_payload(config, task, task_def):
    worker = task["worker"]
    task_def["tags"]["worker-implementation"] = "scriptworker"
    task_def["payload"] = {
        "dryrun": worker["dryrun"],
        "releaseProperties": worker["release-properties"],
        "upstreamArtifacts": worker["upstream-artifacts"],
        "artifactMap": worker["artifact-map"],
    }
