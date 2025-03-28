"""
Utilities for arg parsing in the pipeline.
"""

from typing import Optional


def ensure_string(name: str, value: str) -> str:
    """Ensures that a value exists, handling Taskcluster's "None"."""
    if handle_none_value(value):
        return value
    raise ValueError(f"A value for {name} was not provided.")


def handle_none_value(value: str) -> Optional[str]:
    """When Taskcluster can't find a value, it uses the string "None"."""
    if value == "None":
        return None
    return value
