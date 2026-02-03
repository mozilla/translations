# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


from taskgraph.util.verify import verifications


@verifications.add("full_task_set")
def ensure_no_upload_conflicts(task, taskgraph, scratch_pad, graph_config, parameters):
    """Ensure that each `upload-artifacts` task has a distinct destination.

    This function is called once per task in the graph. `scratch_pad` is shared
    between invocations, making it possible to keep track of state across tasks.
    """
    if task is None or not task.kind.startswith("upload-"):
        return

    # find the destinations the task will publish to
    destinations = []
    if task.kind == "upload-experiment-config":
        for dataUpload in task.task["payload"]["dataMap"]:
            destinations.extend(dataUpload["destinations"])
    else:
        for artifactMap in task.task["payload"]["artifactMap"]:
            for input_paths in artifactMap["paths"].values():
                destinations.extend(input_paths["destinations"])

    for task_ref in destinations:
        # at this point we have `{task-reference: "..."}`
        # dictionaries that contain upload paths with a
        # templated `<decision>` task id in them. this
        # templated form is fine for verification.
        dest = task_ref["task-reference"]
        # final-eval has two tasks: a step of the main pipeline and the one for backfilling
        if dest in scratch_pad and "final-evals" not in dest:
            existing_label = scratch_pad[dest]
            raise Exception(
                f"conflict for upload destination {dest}. {existing_label} and {task.label} would both upload to it!"
            )

        scratch_pad[dest] = task.label
