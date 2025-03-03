#!/usr/bin/env python3

import argparse
import subprocess
import os
import platform
import sys


def get_args():
    parser = argparse.ArgumentParser(description="Run the local docker image.")

    parser.add_argument(
        "--volume",
        action="append",
        help="Specify additional volume(s) to mount in the Docker container.",
        metavar="VOLUME",
    )

    parser.add_argument(
        "--run-as-user",
        action="store_true",
        help="Run the Docker container as the current user's UID and GID.",
    )

    args, other_args = parser.parse_known_args()
    args.other_args = other_args

    return args


def main():
    args = get_args()

    docker_command = [
        "docker",
        "run",
        "--interactive",
        "--tty",
        "--rm",
        "--volume", f"{os.getcwd()}:/builds/worker/checkouts",
        "--workdir", "/builds/worker/checkouts",
        "--expose", "8000", # Expose the mkdocs connection
    ]  # fmt: skip

    # Export the host operating system as an environment variable within the container.
    host_os = platform.system()
    docker_command.extend(["--env", f"HOST_OS={host_os}"])

    # Add additional volumes if provided
    if args.volume:
        for volume in args.volume:
            docker_command.extend(["--volume", volume])

    # Run Docker with the current user's UID and GID if --run-as-user is specified
    if args.run_as_user:
        uid = os.getuid()
        gid = os.getgid()
        docker_command.extend(["--user", f"{uid}:{gid}"])

    # Specify the Docker image
    docker_command.append("translations-local")

    # Append any additional args
    if args.other_args:
        docker_command.extend(args.other_args)

    print("Executing command:", " ".join(docker_command))
    result = subprocess.run(docker_command, check=False)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
