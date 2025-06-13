"""
Uses environment variables to determine if a script is running inside of docker.

python inference/scripts/detect_docker.py task-name
"""

import os
import sys


def print_err(message: str):
    print(message, file=sys.stderr)


def detect_docker(task_name: str | None = None) -> None:
    """
    Uses environment variables to determine if a script is running inside of docker.
    """
    is_docker = os.getenv("IS_DOCKER")
    allow_run_on_host = os.getenv("ALLOW_RUN_ON_HOST")

    if is_docker != "1":
        if allow_run_on_host != "1":
            print_err(
                "\nError: This script needs to be run inside Docker, or you must set ALLOW_RUN_ON_HOST=1."
            )
            if task_name:
                print_err(
                    f"\n Help: To run this script directly in docker, run: task docker-run -- task {task_name}"
                )
            print_err(" Help: To enter docker, run: task docker\n")
            sys.exit(1)
        else:
            print_err("\nALLOW_RUN_ON_HOST is set to 1. Continuing...")


if __name__ == "__main__":
    task_name = sys.argv[1] if len(sys.argv) > 1 else None
    detect_docker(task_name)
