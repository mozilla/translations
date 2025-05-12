"""
This file contains utilities for building and running the TaskGraph in the translations
utilities and tests. This is intended to be run locally.
"""

import json
import os
import requests
import sys
import time
import yaml
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from pathlib import Path
from jsonschema import ValidationError, validate
import taskgraph.parameters

from translations_taskgraph.actions.train import get_config_schema
from taskgraph.config import load_graph_config

ROOT_PATH = (Path(__file__).parent / "../..").resolve()

# The parameters are a read only dict. The class is not exported, so this is a close
# approximation of the type.
Parameters = dict[str, Any]

_last_config_path = None


def get_training_config(cfg_path: str):
    cfg_path = os.path.realpath(cfg_path)
    global _last_config_path  # noqa: PLW0602
    if _last_config_path:
        if cfg_path != _last_config_path:
            raise Exception(
                "Changing the config paths and re-running run_taskgraph is not supported."
            )
        # Don't regenerate the taskgraph for tests, as this can be slow. It's likely that
        # tests will exercise this codepath.
        return

    with open(cfg_path) as f:
        return yaml.safe_load(f)


def run_taskgraph(cfg_path: str, parameters: Parameters) -> None:
    # The callback can be a few standard things like "cancel" and "rerun". Custom actions
    # can be created in taskcluster/translations_taskgraph/actions/ such as the train action.
    callback = "train"

    input = get_training_config(cfg_path)
    if not input:
        # This is probably a test run.
        return

    # This command outputs the stdout. Ignore it here.
    stdout = sys.stdout
    devnull = open(os.devnull, "w")
    sys.stdout = devnull

    # This invokes train_action in taskcluster/translations_taskgraph/actions/train.py
    actions: Any = taskgraph.actions  # type: ignore[reportAttributeAccessIssue]
    actions.trigger_action_callback(
        task_group_id=None,
        task_id=None,
        input=input,
        callback=callback,
        parameters=parameters,
        root="taskcluster",
        test=True,
    )

    sys.stdout = stdout


def get_taskgraph_parameters() -> Parameters:
    # These are required by taskgraph.
    os.environ["TASK_ID"] = "fake_id"
    os.environ["RUN_ID"] = "0"
    os.environ["TASKCLUSTER_ROOT_URL"] = "https://firefox-ci-tc.services.mozilla.com"

    # Load taskcluster/config.yml
    graph_config = load_graph_config("taskcluster")

    # Add the project's taskgraph directory to the python path, and register
    # any extensions present.
    graph_config.register()

    parameters = taskgraph.parameters.load_parameters_file(None, strict=False)
    parameters.check()
    # Example parameters:
    # {
    #   'base_ref': '',
    #   'base_repository': 'git@github.com:mozilla/translations.git',
    #   'base_rev': '',
    #   'build_date': 1704894563,
    #   'build_number': 1,
    #   'do_not_optimize': [],
    #   'enable_always_target': True,
    #   'existing_tasks': {},
    #   'filters': ['target_tasks_method'],
    #   'head_ref': 'main',
    #   'head_repository': 'git@github.com:mozilla/translations.git',
    #   'head_rev': 'e48440fc2c52da770d0f652a32583eae3450766f',
    #   'head_tag': '',
    #   'level': '3',
    #   'moz_build_date': '20240110074923',
    #   'next_version': None,
    #   'optimize_strategies': None,
    #   'optimize_target_tasks': True,
    #   'owner': 'nobody@mozilla.com',
    #   'project': 'translations',
    #   'pushdate': 1704894563,
    #   'pushlog_id': '0',
    #   'repository_type': 'git',
    #   'target_tasks_method': 'default',
    #   'tasks_for': '',
    #   'training_config': { ... },
    #   'version': None
    # }
    return parameters


@dataclass
class TaskgraphFiles:
    # Task label to task description. This is the full task graph for all of the kinds.
    # artifacts/full-task-graph.json
    # { "corpus-merge-parallel-ru-en": TaskDescription, ... }
    full: dict[str, dict[str, Any]]

    # Task id to task description. These are just the tasks that are resolved.
    # artifacts/task-graph.json
    # { "AnmIFVMrT0OoS7UdB4mQGQ": TaskDescription, ... }
    resolved: dict[str, dict[str, Any]]

    @staticmethod
    def from_config(config: str) -> "TaskgraphFiles":
        start = time.time()
        artifacts = Path(__file__).parent / "../../artifacts"

        if os.environ.get("SKIP_TASKGRAPH"):
            print("Using existing taskgraph generation.")
        else:
            print(
                f"Generating the full taskgraph with config {config}, this can take a second. Set SKIP_TASKGRAPH=1 to skip this step."
            )
            run_taskgraph(config, get_taskgraph_parameters())

        with (artifacts / "full-task-graph.json").open() as file:
            full = json.load(file)
        with (artifacts / "task-graph.json").open() as file:
            resolved = json.load(file)

        elapsed_sec = time.time() - start
        print(f"Taskgraph generated in {elapsed_sec:.2f} seconds.")
        return TaskgraphFiles(full=full, resolved=resolved)


# Cache the taskgraphs as it's quite slow to generate them.
_full_taskgraph_cache: dict[str, TaskgraphFiles] = {}


def get_taskgraph_files(config_path: Optional[str] = None) -> TaskgraphFiles:
    """
    Generates the full taskgraph and stores it for re-use. It uses the config.pytest.yml
    in this directory.

    config - A path to a Taskcluster config
    """
    if not config_path:
        config_path = str((ROOT_PATH / "tests/fixtures/config.pytest.yml").resolve())

    if config_path in _full_taskgraph_cache:
        return _full_taskgraph_cache[config_path]

    # Validate the config before using it.
    with open(config_path) as file:
        training_config_yml = yaml.safe_load(file.read())
    with open(ROOT_PATH / "taskcluster/config.yml") as file:
        taskcluster_config_yml = yaml.safe_load(file.read())

    try:
        validate(training_config_yml, get_config_schema(taskcluster_config_yml))
    except ValidationError as error:
        print("Training config:", config_path)
        print(json.dumps(training_config_yml, indent=2))
        raise error

    taskgraph_files = TaskgraphFiles.from_config(config_path)
    _full_taskgraph_cache[config_path] = taskgraph_files
    return taskgraph_files


def get_task_cache_hits() -> Iterable[tuple[str, bool]]:
    """
    Check to see if each task is cached or not, in alphabetical order.
    """

    with (ROOT_PATH / "artifacts/task-graph.json").open() as file:
        task_graph: dict[str, dict[str, Any]] = json.load(file)

    task_ids = list(task_graph.keys())
    task_ids.sort(key=lambda task_id: task_graph[task_id].get("label", task_id))

    for task_id in task_ids:
        task_definition = task_graph[task_id]
        label = task_definition.get("label", task_id)
        index_search = (task_definition.get("optimization") or {}).get("index-search")

        if not index_search:
            yield label, False
            continue

        # Check the first index route only, as there is typically only one.
        index_path = index_search[0]
        url = f"https://firefox-ci-tc.services.mozilla.com/api/index/v1/task/{index_path}"

        timeout_sec = 5.0
        try:
            response = requests.head(url, timeout=timeout_sec)
            if response.status_code == 200:
                yield label, True

            # level-3 is fully trusted, but the caches can still be found from level-1
            # cache sources. Try that as well.
            if "level-3" not in url:
                yield label, False
            response = requests.head(url.replace("level-3", "level-1"), timeout=timeout_sec)
            yield label, response.status_code == 200

        except requests.RequestException:
            # Error contacting the service
            yield label, False
