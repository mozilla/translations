#!/usr/bin/env python3
"""
Publish multiple experiments to Weight & Biases.

Example:
    parse_experiment_dir -d ./tests/data/experiments
"""

import argparse
import logging
import os
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import wandb
import yaml

from translations_parser.data import Metric
from translations_parser.parser import TrainingParser
from translations_parser.publishers import WandB
from translations_parser.utils import extract_dataset_from_tag

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish multiple experiments to Weight & Biases")
    parser.add_argument(
        "--directory",
        "-d",
        help="Path to the experiments directory.",
        type=Path,
        default=Path(Path(os.getcwd())),
    )
    return parser.parse_args()


def parse_experiment(
    project: str,
    group: str,
    name: str,
    logs_file: Path,
    metrics_dir: Path | None = None,
) -> None:
    """
    Parse logs from a Taskcluster dump and publish data to W&B.
    If a metrics directory is set, initially read and publish each `.metrics` values.
    """
    metrics = []
    if metrics_dir:
        for metrics_file in metrics_dir.glob("*.metrics"):
            metrics.append(Metric.from_file(metrics_file, dataset=metrics_file.stem))

    with logs_file.open("r") as f:
        lines = (line.strip() for line in f.readlines())
    parser = TrainingParser(
        lines,
        metrics=metrics,
        publishers=[
            WandB(
                project=project,
                name=name,
                group=group,
            )
        ],
    )
    parser.run()


def publish_group_logs(
    prefix: str,
    project: str,
    group: str,
    existing_runs: list,
) -> None:
    """
    Publish all files within `logs_dir` to W&B artifacts for a specific group.
    A fake W&B run named `group_logs` is created to publish those artifacts among
    with all evaluation files (quantized + experiments).
    """
    logs_dir = Path("/".join([*prefix[:-1], "logs", project, group]))
    # Old experiments use `speed` directory for quantized metrics
    quantized_metrics = sorted(
        Path("/".join([*prefix, project, group, "evaluation", "speed"])).glob("*.metrics")
    )
    evaluation_metrics = sorted((logs_dir / "eval").glob("eval*.log"))
    if quantized_metrics:
        logger.info(f"Found {len(quantized_metrics)} quantized metrics")
    else:
        logger.warning(f"No quantized metric found for group {group}, skipping")
    if evaluation_metrics:
        logger.info(f"Found {len(evaluation_metrics)} evaluation metrics")
    else:
        logger.warning(f"No evaluation metrics not found for group {group}, skipping")

    # Store metrics by run name
    metrics = defaultdict(list)
    # Add "quantized" metrics
    for file in quantized_metrics:
        metrics["quantized"].append(Metric.from_file(file, dataset=file.stem))
    # Add experiment (runs) metrics
    for file in evaluation_metrics:
        model_name, dataset, aug = extract_dataset_from_tag(file.stem)
        with file.open("r") as f:
            lines = f.readlines()
        try:
            metrics[model_name].append(Metric.from_tc_context(dataset, lines))
        except ValueError as e:
            logger.error(f"Could not parse metrics from {file.resolve()}: {e}")

    # Publish missing runs (runs without training data)
    missing_run_metrics = {
        name: metrics for name, metrics in metrics.items() if name not in existing_runs
    }
    for model_name, model_metrics in missing_run_metrics.items():
        logger.info(f"Creating missing run {model_name} with associated metrics")
        publisher = WandB(project=project, name=model_name, group=group)
        publisher.open(TrainingParser(logs_iter=iter([]), publishers=[]))
        publisher.handle_metrics(model_metrics)
        publisher.close()

    # Publication of the `group_logs` fake run
    config = {}
    config_path = Path("/".join([*prefix[:-1], "experiments", project, group, "config.yml"]))
    if not config_path.is_file():
        logger.warning(f"No configuration file at {config_path}, skipping.")
    else:
        # Publish the YAML configuration as configuration on the group run
        with config_path.open("r") as f:
            data = f.read()
        try:
            config.update(yaml.safe_load(data))
        except Exception as e:
            logger.error(f"Config could not be read at {config_path}: {e}")

    publisher = WandB(
        project=project,
        group=group,
        name="group_logs",
        notes=(
            "Experiments summary for the group.\n"
            "The configuration section contains `config.yaml` values, logs "
            "are uploaded as artifacts and all metrics are reported in a table."
        ),
    )
    publisher.wandb = wandb.init(
        project=project,
        group=group,
        name="group_logs",
        config=config,
    )
    if metrics:
        # Publish all evaluation metrics to a table
        table = wandb.Table(
            columns=["Group", "Model", "Dataset", "BLEU", "chrF"],
            data=[
                [group, run_name, metric.dataset, metric.bleu_detok, metric.chrf]
                for run_name, run_metrics in metrics.items()
                for metric in run_metrics
            ],
        )
        publisher.wandb.log({"metrics": table})
    if logs_dir.is_dir():
        # Publish logs directory content as artifacts
        artifact = wandb.Artifact(name=group, type="logs")
        artifact.add_dir(local_path=str(logs_dir.resolve()))
        publisher.wandb.log_artifact(artifact)

    publisher.wandb.finish()


def main() -> None:
    args = get_args()
    directory = args.directory
    # Ignore files with a different name than "train.log"
    file_groups = {
        path: list(files)
        for path, files in groupby(
            sorted(directory.glob("**/train.log")), lambda path: path.parent
        )
    }
    logger.info(f"Reading {len(file_groups)} train.log data")
    prefix = os.path.commonprefix([path.parts for path in file_groups])
    if "models" in prefix:
        prefix = prefix[: prefix.index("models") + 1]

    last_index = None
    existing_runs = []
    for index, (path, files) in enumerate(file_groups.items(), start=1):
        logger.info(f"Parsing folder {path.resolve()}")
        parents = path.parts[len(prefix) :]
        if len(parents) < 3:
            logger.warning(f"Skipping folder {path.resolve()}: Unexpected folder structure")
            continue
        project, group, *name = parents
        base_name = name[0]
        name = "_".join(name)

        # Publish a run for each file inside that group
        for file in files:
            # Also publish metric files when available
            metrics_path = Path("/".join([*prefix, project, group, "evaluation", base_name]))
            metrics_dir = metrics_path if metrics_path.is_dir() else None
            if metrics_dir is None:
                logger.warning("Evaluation metrics files not found, skipping.")
            try:
                parse_experiment(project, group, name, file, metrics_dir=metrics_dir)
                existing_runs.append(name)
            except Exception as e:
                logger.error(f"An exception occured parsing {file}: {e}")

        # Try to publish related log files to the group on a last run named "group_logs"
        if index == len(file_groups) or last_index and last_index != (project, group):
            last_project, last_group = (
                last_index
                if last_index
                # May occur when handling a single run
                else (project, group)
            )
            logger.info(
                f"Publishing '{last_project}/{last_group}' evaluation metrics and files (fake run 'group_logs')"
            )
            publish_group_logs(prefix, last_project, last_group, existing_runs)
            existing_runs = []
        last_index = (project, group)
