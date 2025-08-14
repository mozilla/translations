"""
Trigger a training task from the CLI on your current branch.

For example:

  task train -- --config configs/experiments-2024-H2/en-lt-experiments-2024-H2-hplt-nllb.yml
"""

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
from time import sleep
from typing import Any, Optional, Tuple
from github import Github
import os
import yaml
import jsone
from taskgraph.util.taskcluster import get_artifact
from taskcluster import Hooks
from taskcluster.helper import TaskclusterConfig

ROOT_URL = "https://firefox-ci-tc.services.mozilla.com"


def run(command: list[str], env={}):
    return subprocess.run(
        command, capture_output=True, check=True, text=True, env={**os.environ, **env}
    ).stdout.strip()


def check_if_pushed(branch: str) -> bool:
    try:
        remote_commit = run(["git", "rev-parse", f"origin/{branch}"])
        local_commit = run(["git", "rev-parse", branch])
        return local_commit == remote_commit
    except subprocess.CalledProcessError:
        return False


def get_decision_task_push(branch: str):
    g = Github()
    repo_name = "mozilla/translations"
    print(f'Looking up "{repo_name}"')
    repo = g.get_repo(repo_name)
    ref = f"heads/{branch}"

    print('Finding the "Decision Task (push)"')
    checks = repo.get_commit(ref).get_check_runs()
    decision_task = None
    for check in checks:
        if check.name == "Decision Task (push)":
            decision_task = check

    return decision_task


def get_task_id_from_url(task_url: str):
    """
    Extract the task id from a task url

    e.g. https://firefox-ci-tc.services.mozilla.com/tasks/PhAMJTZBSmSeWStXbR72xA
         returns
         "PhAMJTZBSmSeWStXbR72xA"
    """
    return task_url.split("/")[-1]


def get_train_action(decision_task_id: str):
    actions_json = get_artifact(decision_task_id, "public/actions.json")

    for action in actions_json["actions"]:
        if action["name"] == "train":
            return action

    print("Could not find the train action.")
    print(actions_json)
    sys.exit(1)


def trigger_training(decision_task_id: str, config: dict[str, Any]) -> Optional[str]:
    taskcluster = TaskclusterConfig(ROOT_URL)
    taskcluster.auth()
    hooks: Hooks = taskcluster.get_service("hooks")
    train_action = get_train_action(decision_task_id)

    # Render the payload using the jsone schema.
    hook_payload = jsone.render(
        train_action["hookPayload"],
        {
            "input": config,
            "taskId": None,
            "taskGroupId": decision_task_id,
        },
    )

    start_stage: str = config["target-stage"]
    if start_stage.startswith("train"):
        evaluate_stage = start_stage.replace("train-", "evaluate-")
        red = "\033[91m"
        reset = "\x1b[0m"
        print(
            f'\n{red}WARNING:{reset} target-stage is "{start_stage}", did you mean "{evaluate_stage}"'
        )

    confirmation = input("\nStart training? [Y,n]\n")
    if confirmation and confirmation.lower() != "y":
        return None

    # https://docs.taskcluster.net/docs/reference/core/hooks/api#triggerHook
    response: Any = hooks.triggerHook(
        train_action["hookGroupId"], train_action["hookId"], hook_payload
    )

    action_task_id = response["status"]["taskId"]

    print(f"Train action triggered: {ROOT_URL}/tasks/{action_task_id}")

    return action_task_id


def validate_taskcluster_credentials():
    try:
        run(["taskcluster", "--help"])
    except Exception:
        print("The taskcluster client library must be installed on the system.")
        print("https://github.com/taskcluster/taskcluster/tree/main/clients/client-shell")
        sys.exit(1)

    if not os.environ.get("TASKCLUSTER_ACCESS_TOKEN"):
        print("You must log in to Taskcluster. Run the following:")
        print(f'eval `TASKCLUSTER_ROOT_URL="{ROOT_URL}" taskcluster signin`')
        sys.exit(1)

    try:
        run(
            [
                "taskcluster",
                "signin",
                "--check",
            ],
            {"TASKCLUSTER_ROOT_URL": ROOT_URL},
        )
    except Exception:
        print("Your Taskcluster credentials have expired. Run the following:")
        print(f'eval `TASKCLUSTER_ROOT_URL="{ROOT_URL}" taskcluster signin`')
        sys.exit(1)


def log_config_info(config_path: Path, config: dict):
    print(f"\nUsing config: {config_path}\n")

    experiment = config["experiment"]
    config_details: list[Tuple[str, Any]] = []
    config_details.append(("experiment.name", experiment["name"]))
    config_details.append(("experiment.src", experiment["src"]))
    config_details.append(("experiment.trg", experiment["trg"]))
    if config.get("start-task-prefix"):
        config_details.append(("start-task-prefix", config["start-task-prefix"]))
    config_details.append(("target-stage", config["target-stage"]))

    previous_group_ids = config.get("previous-group-ids")
    if previous_group_ids:
        config_details.append(("previous-group-ids", previous_group_ids))

    continuation: Optional[dict] = config.get("continuation")
    if continuation:
        config_details.append(("continuation", json.dumps(continuation, indent=2)))

    key_len = 0
    for key, _ in config_details:
        key_len = max(key_len, len(key))

    for key, value in config_details:
        if "\n" in value:
            # Nicely indent any multiline value.
            padding = " " * (key_len + 6)
            lines = [padding + n for n in value.split("\n")]
            value = "\n".join(lines).strip()  # noqa: PLW2901

        print(f"{key.rjust(key_len + 4, ' ')}: {value}")


def write_to_log(config_path: Path, config: dict, action_task_id: str, branch: str):
    """
    Persist the training log to disk.
    """
    training_log = Path(__file__).parent / "../trigger-training.log"
    experiment = config["experiment"]
    git_hash = run(["git", "rev-parse", "--short", branch]).strip()

    with open(training_log, "a") as file:
        lines = [
            "",
            f"config: {config_path}",
            f"name: {experiment['name']}",
            f"langpair: {experiment['src']}-{experiment['trg']}",
            f"time: {datetime.datetime.now()}",
            f"train action: {ROOT_URL}/tasks/{action_task_id}",
            f"branch: {branch}",
            f"hash: {git_hash}",
        ]
        for line in lines:
            file.write(line + "\n")


def get_decision_task_push_loop(branch: str):
    timeout = 20
    while True:
        decision_task = get_decision_task_push(branch)

        if decision_task:
            if decision_task.status == "completed" and decision_task.conclusion == "success":
                # The decision task is completed.
                break
            elif decision_task.status == "queued":
                print(f"Decision task is queued, trying again in {timeout} seconds")
            elif decision_task.status == "in_progress":
                print(f"Decision task is in progress, trying again in {timeout} seconds")
            else:
                # The task failed.
                print(
                    f'Decision task is "{decision_task.status}" with the conclusion "{decision_task.conclusion}"'
                )
                sys.exit(1)
        else:
            print(f"Decision task is not available, trying again in {timeout} seconds")

        sleep(timeout)

    return decision_task


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        # Preserves whitespace in the help text.
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--config", type=Path, required=True, help="Path the config")
    parser.add_argument(
        "--branch",
        type=str,
        required=False,
        help="The name of the branch, defaults to the current branch",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the checks for the branch being up to date",
    )
    parser.add_argument(
        "--no_interactive",
        action="store_true",
        help="Skip the confirmation",
    )

    args = parser.parse_args()
    branch = args.branch

    validate_taskcluster_credentials()

    if branch:
        print(f"Using --branch: {branch}")
    else:
        branch = run(["git", "branch", "--show-current"])
        print(f"Using current branch: {branch}")

    if branch != "main" and not branch.startswith("dev") and not branch.startswith("release"):
        print(f'The git branch "{branch}" must be "main", or start with "dev" or "release"')
        sys.exit(1)

    if check_if_pushed(branch):
        print(f"Branch '{branch}' is up to date with origin.")
    elif args.force:
        print(
            f"Branch '{branch}' is not fully pushed to origin, bypassing this check because of --force."
        )
    else:
        print(
            f"Error: Branch '{branch}' is not fully pushed to origin. Use --force or push your changes."
        )
        sys.exit(1)

    if branch != "main" and not branch.startswith("dev") and not branch.startswith("release"):
        print(
            f"Branch must be `main` or start with `dev` or `release` for training to run. Detected branch was {branch}"
        )

    decision_task = get_decision_task_push_loop(branch)
    decision_task_id = get_task_id_from_url(decision_task.details_url)

    with args.config.open() as file:
        config: dict = yaml.safe_load(file)

    log_config_info(args.config, config)
    action_task_id = trigger_training(decision_task_id, config)
    if action_task_id:
        write_to_log(args.config, config, action_task_id, branch)


if __name__ == "__main__":
    main()
