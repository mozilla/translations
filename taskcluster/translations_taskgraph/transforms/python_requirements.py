# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from taskgraph.transforms.base import TransformSequence
from taskgraph.util.schema import Schema
from voluptuous import ALLOW_EXTRA, Any, Optional


SCHEMA = Schema(
    {
        Optional("python-requirements"): Any(str, [str]),
    },
    extra=ALLOW_EXTRA,
)

transforms = TransformSequence()
transforms.add_validate(SCHEMA)


@transforms.add
def install_python_requirements(config, jobs):
    for job in jobs:
        requirements = job.pop("python-requirements", [])
        if not requirements:
            yield job
            continue

        if isinstance(requirements, str):
            requirements = [requirements]

        command = job.get("run", {}).get("command")
        if not (
            isinstance(command, list)
            and len(command) == 3
            and command[:2] == ["bash", "-c"]
            and isinstance(command[2], str)
        ):
            raise Exception(
                "python-requirements requires run.command to use ['bash', '-c', <script>]"
            )

        activation = "source .venv/bin/activate &&"
        if activation not in command[2]:
            raise Exception(
                "python-requirements requires run.command to activate .venv first"
            )

        installs = " ".join(
            f"uv pip install -r {requirements_file} &&"
            for requirements_file in requirements
        )
        command[2] = command[2].replace(
            activation,
            f"{activation} {installs}",
            1,
        )

        yield job
