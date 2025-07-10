# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This transform is responsible for resolving the `worker.bucket` key, which is
typically keyed by `tasks-for`.
"""

from taskgraph.transforms.base import TransformSequence
from taskgraph.util.schema import resolve_keyed_by

transforms = TransformSequence()


@transforms.add
def evaluate_keyed_by(_, jobs):
    for job in jobs:
        # `tasks-for` is a key that is always available; it does not need
        # to be specified explicitly.
        resolve_keyed_by(
            job["worker"],
            "bucket",
            item_name=job["description"],
        )

        yield job
