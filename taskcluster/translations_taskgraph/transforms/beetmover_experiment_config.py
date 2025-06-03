# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from base64 import b64encode
import json

from taskgraph.transforms.base import TransformSequence
from taskgraph.util.schema import resolve_keyed_by

transforms = TransformSequence()


@transforms.add
def add_training_config(config, jobs):
    for job in jobs:
        training_config = json.dumps(config.params["training_config"])
        job["worker"]["data-map"][0]["data"] = b64encode(training_config.encode()).decode()

        yield job


@transforms.add
def evaluate_keyed_by(config, jobs):
    for job in jobs:
        resolve_keyed_by(
            job["worker"],
            "bucket",
            item_name=job["description"],
        )

        yield job
