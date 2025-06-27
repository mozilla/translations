"""
Common utilities related to working with Marian.
"""

from logging import Logger
import os
import sys
import subprocess
from pathlib import Path
from typing import Union

from pipeline.common.logging import get_logger

import yaml


def get_combined_config(config_path: Path, extra_marian_args: list[str]) -> dict[str, any]:
    """
    Frequently we combine a Marian yml config with extra marian args when running
    training. To get the final value, add both here.
    """
    return {
        **yaml.safe_load(config_path.open()),
        **marian_args_to_dict(extra_marian_args),
    }


def marian_args_to_dict(extra_marian_args: list[str]) -> dict[str, Union[str, bool, list[str]]]:
    """
    Converts marian args, to the dict format. This will combine a decoder.yml
    and extra marian args.

    e.g. `--precision float16` becomes {"precision": "float16"}
    """
    decoder_config = {}
    if extra_marian_args and extra_marian_args[0] == "--":
        extra_marian_args = extra_marian_args[1:]

    previous_key = None
    for arg in extra_marian_args:
        if arg.startswith("--"):
            previous_key = arg[2:]
            decoder_config[previous_key] = True
            continue

        if not previous_key:
            raise Exception(
                f"Expected to have a previous key when converting marian args to a dict: {extra_marian_args}"
            )

        prev_value = decoder_config.get(previous_key)
        if prev_value is True:
            decoder_config[previous_key] = arg
        elif isinstance(prev_value, list):
            prev_value.append(arg)
        else:
            decoder_config[previous_key] = [prev_value, arg]

    return decoder_config


def assert_gpus_available(logger: Logger = get_logger("gpu_check")) -> None:
    """
    Sometimes the GPUs aren't available when running tasks on GPU machines in the cloud.
    This function reports on the GPUs available, and exits the task with an
    EX_TEMPFAIL (75) exit code when the machines are not available. Taskcluster can
    restart the tasks via the `retry-exit-status` property.
    """

    if "USE_CPU" in os.environ or "COMET_CPU" in os.environ:
        return

    query = {
        "name": "Name",
        "driver_version": "Driver Version",
        "vbios_version": "GPU BIOS",
        "memory.total": "Memory Total",
        "memory.free": "Memory Free",
        "compute_cap": "Compute Capability (https://developer.nvidia.com/cuda-gpus)",
        "temperature.gpu": "GPU temperature (Celsius)",
    }

    fields = list(query.keys())

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        raise Exception(
            "nvidia-smi not found. Ensure NVIDIA drivers are installed and nvidia-smi is in PATH."
        )

    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        logger.error(f"nvidia-smi failed with return code {result.returncode}")
        for line in stdout.splitlines():
            logger.error(f"[nvidia-smi(stdout)] {line}")
        for line in stderr.splitlines():
            logger.error(f"[nvidia-smi(stderr)] {line}")
        logger.error("No GPUs were found available on this machine. Exiting with EX_TEMPFAIL (75)")
        sys.exit(75)

    output = result.stdout.strip()
    if not output:
        logger.info("No GPUs found by nvidia-smi, exiting EX_TEMPFAIL (75).")
        sys.exit(75)

    logger.info("CUDA-capable GPU(s) detected.")
    for idx, line in enumerate(output.splitlines()):
        values = [v.strip() for v in line.split(",")]
        logger.info(f"GPU {idx}:")
        for key, value in zip(query.values(), values):
            logger.info(f"  {key}: {value}")
        logger.info("")
