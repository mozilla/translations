# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest

from translations_taskgraph.transforms.python_requirements import (
    install_python_requirements,
)


class TestPythonRequirements(unittest.TestCase):
    def test_installs_requirements_after_virtualenv_activation(self):
        job = {
            "python-requirements": [
                "$VCS_PATH/pipeline/eval/requirements/eval.txt",
                "$VCS_PATH/pipeline/extra/requirements/extra.txt",
            ],
            "run": {
                "command": [
                    "bash",
                    "-c",
                    "uv venv --system-site-packages && "
                    "source .venv/bin/activate && python script.py",
                ]
            },
        }

        [transformed] = list(install_python_requirements(None, [job]))

        self.assertNotIn("python-requirements", transformed)
        self.assertEqual(
            transformed["run"]["command"][2],
            "uv venv --system-site-packages && source .venv/bin/activate && "
            "uv pip install -r $VCS_PATH/pipeline/eval/requirements/eval.txt && "
            "uv pip install -r $VCS_PATH/pipeline/extra/requirements/extra.txt && "
            "python script.py",
        )

    def test_requires_virtualenv_activation(self):
        job = {
            "python-requirements": "requirements.txt",
            "run": {"command": ["bash", "-c", "python script.py"]},
        }

        with self.assertRaisesRegex(Exception, "activate .venv"):
            list(install_python_requirements(None, [job]))


if __name__ == "__main__":
    unittest.main()
