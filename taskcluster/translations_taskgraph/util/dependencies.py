# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


def get_upload_artifacts_upstream_dependency(dependencies):
    """Sanity checks upload-artifacts upstream dependencies and returns the name and
    label of the task that the task will upload files from.

    In typical use cases, upload-artifact tasks will have one upstream dependency
    because they will be downstream of another newly created task before
    running. In the rare case where their non-decision upstream is already cached
    they will have both that task and the decision task as an upstream.

    (There are no valid configurations where an upload-artifacts task would have
    0 or more than 2 dependencies.)"""

    keys = list(dependencies.keys())
    match len(dependencies):
        case 0:
            raise Exception("upstream dependency is required for upload-artifacts tasks!")
        case 1:
            if keys[0] == "decision":
                raise Exception(
                    f"upload-artifacts tasks must have a non-decision task dependency! found: {dependencies}"
                )
            return keys[0], dependencies[keys[0]]

        case 2:
            if "decision" not in keys:
                raise Exception(
                    f"upload-artifacts tasks must only have one non-decision task dependency! found: {dependencies}"
                )

            keys.remove("decision")
            return keys[0], dependencies[keys[0]]

    raise Exception(f"upload-artifacts task has too many upstreams! found: {dependencies}")
