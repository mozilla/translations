#!/usr/bin/env python3

import argparse
import subprocess
import os
import platform
import sys

CONTAINER_NAME = "translations-local-dev"
IMAGE_NAME = "translations-local"


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

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove any existing container before starting a new one.",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=CONTAINER_NAME,
        help="Provide a custom container name",
    )

    return parser.parse_known_args()


def container_exists(container_name: str):
    command = [
        "docker", "ps",
        "--all",
        "--filter", f"name={container_name}",
        "--format", "{{.Names}}"
    ]  # fmt: skip
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    return container_name in result.stdout.splitlines()


def main() -> None:
    args, run_in_docker = get_args()

    container_name: str = args.name

    # Use a temporary container when running an arbitrary command in docker.
    use_temp_container = run_in_docker and run_in_docker != ["bash"]
    if use_temp_container:
        print("[docker-run] Using a temporary container")
        # Run in a temporary container if other args are present.
        container_name = f"{container_name}-tmp"

    if args.clean and container_exists(container_name):
        print(f'[docker-run] Removing the "{container_name}" container')
        subprocess.run(["docker", "container", "rm", "--force", container_name], check=True)

    if container_exists(container_name):
        print(f"[docker-run] Attaching to the existing container {container_name}")
        result = subprocess.run(
            ["docker", "container", "start", "--attach", "--interactive", container_name],
            check=False,
        )
    else:
        # Create and run the container.
        docker_command = [
            "docker", "container", "run",
            "--interactive",
            "--tty",
            "--name", container_name,
            "--volume", f"{os.getcwd()}:/builds/worker/checkouts",
            "--workdir", "/builds/worker/checkouts",
            "--publish", "8000:8000",
            "--publish", "8080:8080"
        ]  # fmt: skip

        # Export the host operating system as an environment variable within the container.
        host_os = platform.system()
        docker_command.extend(["--env", f"HOST_OS={host_os}"])

        if run_in_docker:
            # We're just running a command an executing, remove the temporary container
            # after creating it.
            docker_command.append("--rm")

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
        docker_command.append(IMAGE_NAME)

        # Append any additional args
        docker_command.extend(run_in_docker)

        print("[docker-run] Running:", " ".join(docker_command))
        result = subprocess.run(docker_command, check=False)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
