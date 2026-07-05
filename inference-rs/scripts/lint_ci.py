#!/usr/bin/env python3
"""
Guard against `task rs:check` and the GitHub CI workflow drifting apart.

`scripts/check.py` (CHECKS) is the single source of truth for what `task rs:check`
runs. Each of those checks should have a matching job in the CI workflow that runs
the same `task rs:<name>` — except for a small CI-exempt set (heavy builds we keep
in local dev / taskcluster). This asserts that invariant so adding a check in one
place forces the other, and fails loudly with the specific difference otherwise.

Pure stdlib (like check.py) so it needs no poetry env to run. Invoked via
`task rs:lint-ci`, and itself listed in CHECKS so it runs under `task rs:check`.
"""

import importlib.util
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
WORKFLOW = REPO_ROOT / ".github/workflows/inference-rs.yml"

# Checks in CHECKS that intentionally do NOT run in CI. inference-build is the C++
# marian reference engine — a ~20-40 min build needing the full C++ toolchain and
# submodules; it stays in local dev / taskcluster, not this workflow.
CI_EXEMPT = {"inference-build"}


def load_checks() -> list:
    """Import CHECKS from the sibling check.py without running its main()."""
    spec = importlib.util.spec_from_file_location("rs_check", SCRIPTS_DIR / "check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CHECKS


def workflow_tasks(text: str) -> set:
    """The set of `task rs:<name>` invocations declared in the workflow.

    Skips full-line YAML comments so prose mentions of `task rs:check` (the
    aggregator, never its own job) don't count, and drops rs:check itself.
    """
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    tasks = set(re.findall(r"\btask\s+(rs:[A-Za-z0-9-]+)", "\n".join(lines)))
    tasks.discard("rs:check")
    return tasks


def main() -> None:
    if not WORKFLOW.exists():
        sys.exit(f"CI workflow not found at {WORKFLOW.relative_to(REPO_ROOT)}")

    expected = {check["task"] for check in load_checks()} - CI_EXEMPT
    actual = workflow_tasks(WORKFLOW.read_text())

    missing_in_ci = expected - actual  # in rs:check, no CI job
    missing_in_checks = actual - expected  # a CI job, not in rs:check (or wrongly exempt)

    if not missing_in_ci and not missing_in_checks:
        print(
            f"CI jobs match rs:check ({len(expected)} checks; exempt: {', '.join(sorted(CI_EXEMPT))})."
        )
        return

    print("CI and `task rs:check` have drifted:\n")
    if missing_in_ci:
        print("  In rs:check (scripts/check.py) but no CI job runs it:")
        for task in sorted(missing_in_ci):
            print(f"    - {task}")
        print(
            f"  Add a job running `{sorted(missing_in_ci)[0]}` to {WORKFLOW.relative_to(REPO_ROOT)},"
        )
        print("  or add it to CI_EXEMPT here if it should not run in CI.\n")
    if missing_in_checks:
        print("  Run by a CI job but not in rs:check (scripts/check.py CHECKS):")
        for task in sorted(missing_in_checks):
            print(f"    - {task}")
        print("  Add it to CHECKS, or remove the CI job.\n")
    sys.exit(1)


if __name__ == "__main__":
    main()
