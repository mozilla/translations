---
layout: default
title: Development
nav_order: 8
---

# Development

## Configuring development environment

Set up a local [poetry](https://python-poetry.org/) environment.

Install [Taskfile](https://taskfile.dev/installation/) to run local scripts. 
[Taskfile.yml](https://github.com/mozilla/translations/blob/main/Taskfile.yml) is similar to a Makefile but provides a better support for command line interface.

Tasks can be listed by running `task --list-all`.

## Running tests

To run unit tests locally:
```bash
task test
```

To run a specific test pass pytest arguments to the `test` command:

```bash
task test -- tests/test_alignments.py
```

Some tests require a compiled Marian and other tools, which is easier to set up with Docker.

It's also possible to run simpler tests using regular `pytest` commands if all the requirements are installed locally.

## Docker

Build a Docker image and attach to the running container with:

```bash
task docker
```

Then run commands from inside the container in the same way as from the local environment, for example:

```bash
task test
```

## Updating Python packages

Each pipeline step has its own requirements file with the required packages 
and a compiled full list of packages generated with `pip-compile` from `pip-tools` (requires `pip install pip-tools`).

For example for the [evaluation step](https://github.com/mozilla/translations/tree/main/pipeline/eval/requirements):
```
pipeline/eval/requirements/eval.in
pipeline/eval/requirements/eval.txt
```


To update the packages, first update the `eval.in` file, then run:

```bash
task update-requirements -- pipeline/eval/requirements/eval.in
```


## Before committing

Make sure to run linting with `task lint-fix`.

For changes in the Taskcluster graph run `task taskgraph-validate` to validate the graph locally.


## CI

We run all training pipeline steps with a minimal config on pull requests. It runs on the same hardware as a production run.
Make sure to use `[skip ci]` directive in the PR description not to trigger the run if it's not intended to save resources.
If you do run it, minimize pushing to the branch.

Ideally every new push to PR without `[skip ci]` should mean to test the new changes using CI.

We do not run the pipeline on branches without a corresponding pull request.

## Architecture

All the pipeline steps are independent and contain scripts that accept arguments, read input files from disk and output the results to disk.
It allows writing the steps in any language (currently it's historically mostly bash and Python) and
represent the pipeline as a directed acyclic graph (DAG).

The DAG of tasks can be launched using any workflow manager
(currently we support only [Taskcluster](../infrastructure/task-cluster.md). [Snakemake](../infrastructure/snakemake.md) integration is unmaintained, but we accept contributions).
The workflow manager integration code should not include any training specific logic but rather implement it as a script
in the `pipeline` directory.

## Conventions

- Scripts inside the `pipeline` directory are independent and operate only using input arguments, input files
  and global environment variables.

- All scripts test expected environment variables early.

- If a script step fails, it can be safely retried.

- Ideally, every script should start from the last unfinished step,
  checking presence of intermediate results of previous steps. 
  For example a model training script should utilize the last training checkpoint.

- A script fails as early as possible.

- Maximum bash verbosity is set for easy debugging.

- Input data is always read only.

- Output data is placed in a new folder for script results.

- It is expected that the specified output folder might not exist and should be created by the script.

- A script creates a folder for intermediate files and cleans it in the end
  unless intermediate files are useful for retries.

- Global bash variables are upper case, local variables are lower case.

- Scripts should utilize resources provided by a workflow manager (for example number of threads).

- If the logic of the script is getting more complex, it should be written in Python since it can be easier to maintain

- Python scripts should use named arguments and argparse functionality
