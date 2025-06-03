# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""
This transform is responsible for fishing the `training_config` out of parameters
and injecting it into the `data-map` used by the `upload-experiment-config` task.
"""

from base64 import b64encode
import json

from taskgraph.transforms.base import TransformSequence

transforms = TransformSequence()


@transforms.add
def add_training_config(config, jobs):
    for job in jobs:
        training_config = json.dumps(config.params["training_config"])
        job["worker"]["data-map"][0]["data"] = b64encode(training_config.encode()).decode()

        yield job
