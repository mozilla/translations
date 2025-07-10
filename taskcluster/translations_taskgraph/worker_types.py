# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from taskgraph.transforms.task import payload_builder, taskref_or_string
from voluptuous import Required

from translations_taskgraph.util.dependencies import get_upload_artifacts_upstream_dependency


@payload_builder(
    "scriptworker-beetmover-translations",
    schema={
        Required("app-name"): str,
        Required("bucket"): str,
        # list of upstream artifacts and/or globs
        Required("upstream-artifacts"): [str],
        # artifact names or globs / list of destinations
        Required("artifact-map"): {str: [taskref_or_string]},
    },
)
def build_upload_artifacts_payload(_, task, task_def):
    """Converts the simplified `worker` from an upload-artifacts task into a valid,
    runnable payload, and adds the appropriate scopes and tags."""

    # Most upload-artifacts tasks have a pipeline task as their upstream. When
    # no dependencies are listed, assume the upstream is the decision task is
    # the upstream, such as for `upload-task-graph`.
    if task.get("dependencies"):
        upstream_task = get_upload_artifacts_upstream_dependency(task["dependencies"])[0]
    else:
        upstream_task = "decision"

    worker = task["worker"]
    task_def["tags"]["worker-implementation"] = "scriptworker"
    task_def["payload"] = {
        "releaseProperties": {"appName": worker["app-name"]},
        "upstreamArtifacts": [
            {
                "paths": worker["upstream-artifacts"],
                "taskId": {"task-reference": f"<{upstream_task}>"},
                "taskType": "build",
                "optional": True,
            }
        ],
        "artifactMap": [{"taskId": {"task-reference": f"<{upstream_task}>"}, "paths": {}}],
    }
    for pattern, dests in worker["artifact-map"].items():
        task_def["payload"]["artifactMap"][0]["paths"][pattern] = {"destinations": dests}

    task_def["scopes"] = [
        f"project:translations:releng:beetmover:bucket:{worker['bucket']}",
        "project:translations:releng:beetmover:action:upload-translations-artifacts",
    ]
