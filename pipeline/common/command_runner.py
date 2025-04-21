import os
import re
from shlex import join
import shlex
import subprocess
import sys
import traceback
from typing import Optional


def _get_indented_command_string(command_parts: list[str]) -> str:
    """
    Print out a command with the flags indented, so that it's easy to read.
    """
    command = join(command_parts)
    parts = re.split(r"( --\w)", command)

    formatted_command = [parts[0].strip()]

    for i in range(1, len(parts), 2):
        option = parts[i].strip() + parts[i + 1].strip()
        formatted_command.append(f"  {option}")

    return "\n".join(formatted_command)


def apply_command_args(dict: dict[str, any]):
    """
    Takes in a dictionary, and applies the keys as command line flags.

    input: { "key": "value" }
    output: "--key value"

    input: { "inputs": ["valueA", "valueB"] }
    output: "--inputs valueA valueB"
    """

    for key, value in dict.items():
        yield f"--{key}"
        if value is None:
            continue

        if isinstance(value, (list, tuple)):
            for v in value:
                yield str(v)
            continue

        yield str(value)


def run_command_pipeline(
    commands: list[list[str]], pipe_stderr=False, capture=False, logger=None
) -> str | None:
    """
    Executes a series of shell commands in a pipeline, where the output of one command
    is piped to the next. Optionally captures the final output or logs the pipeline
    process. It raises `CalledProcessError` if any command in the pipeline fails.

    Args:
      commands: A list of command arguments where each command is
        represented as a list of strings.
      pipe_stderr: If True, pipes `stderr` of each command into `stdout`.
      capture: If True, captures and returns the output of the final command in the
        pipeline. If False, output is printed to stdout. Defaults to False.
      logger: A logger instance used for logging the command execution. If provided,
        it will log the constructed pipeline commands. Defaults to None.

    Example:
      python_scripts = run_command_pipeline(
        [
            ["ls", "-l"],
            ["grep", ".py"],
            ["sort"]
        ],
        capture=True
      )
    """
    if pipe_stderr:
        joiner = "2>&1 |"
    else:
        joiner = "|"

    if logger:
        # Log out a nice representation of this command.
        final_command = _get_indented_command_string(commands[0])
        for command_parts in commands[1:]:
            final_command = (
                f"{final_command}\n{joiner} {_get_indented_command_string(command_parts)}"
            )

        logger.info("Running:")
        for line in final_command.split("\n"):
            logger.info(line)

    command_string = f" {joiner} ".join([shlex.join(command) for command in commands])

    process = subprocess.Popen(
        command_string,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if pipe_stderr else None,
        # 1 means line buffered when used with text=True.
        bufsize=1,
        text=True,
    )

    output_lines = []
    error_code: Optional[int] = None
    try:
        assert process.stdout is not None
        for line in process.stdout:
            # Stream to stdout.
            if logger:
                logger.info(line.rstrip())
            else:
                print(line, end="")

            if capture:
                output_lines.append(line)

            # Handle known errors where the task could be restarted.
            if "Curand error 203" in line:
                # This is an issue with the GPU not being available, and the task failing.
                # A restart could fix it.
                error_code = 75  # EX_TEMPFAIL
        if capture:
            return "".join(output_lines)
        return None
    except Exception as exception:
        # Ensure the process is killed if an unexpected error occurs.
        process.kill()
        raise exception
    finally:
        process.wait()
        if not error_code:
            error_code = process.returncode

        if error_code != 0:
            message = f"Command '{command_string}' returned non-zero exit status {error_code}."
            if logger:
                logger.error(message)
            else:
                print(message)

            traceback.print_stack()
            sys.exit(error_code or process.returncode)


def run_command(
    command: list[str], capture=False, shell=False, logger=None, env=None
) -> str | None:
    """
    Runs a command and outputs a nice representation of the command to a logger, if supplied.

    Args:
      command: The command arguments provided to subprocess.check_call
      capture: If True, captures and returns the output of the final command in the
        pipeline. If False, output is printed to stdout.
      logger: A logger instance used for logging the command execution. If provided,
        it will log the pipeline commands.
      env: The environment object.

    Example:
      directory_listing = run_command(
        ["ls", "-l"],
        capture=True
      )
    """
    # Expand any environment variables.
    command = [os.path.expandvars(part) for part in command]

    if logger:
        # Log out a nice representation of this command.
        logger.info("Running:")
        for line in _get_indented_command_string(command).split("\n"):
            logger.info(line)

    if capture:
        return subprocess.check_output(command).decode("utf-8")

    subprocess.check_call(command, env=env)
