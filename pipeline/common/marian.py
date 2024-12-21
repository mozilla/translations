"""
Common utilities related to working with Marian.
"""

from pathlib import Path
from typing import Union

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
