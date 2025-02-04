import csv
import logging
import sys
from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import wandb
import yaml

from translations_parser.data import Metric, TrainingEpoch, ValidationEpoch
from translations_parser.utils import parse_task_label, parse_gcp_metric, patch_model_name

logger = logging.getLogger(__name__)

METRIC_KEYS = sorted(set(Metric.__annotations__.keys()) - {"importer", "dataset", "augmentation"})


class Publisher(ABC):
    """
    Abstract class used to publish parsed data.

    Either the `handle_*` methods can be overriden for real time
    publication (introduced later on) or the `publish` method
    with all results (including parser run date, configuration…).
    """

    def open(self, parser) -> None:
        ...

    def handle_training(self, training: TrainingEpoch) -> None:
        ...

    def handle_validation(self, validation: ValidationEpoch) -> None:
        ...

    def handle_metrics(self, metrics: Sequence[Metric]) -> None:
        ...

    def publish(self) -> None:
        ...

    def close(self) -> None:
        ...


class CSVExport(Publisher):
    def __init__(self, output_dir: Path) -> None:
        from translations_parser.parser import TrainingParser

        if not output_dir.is_dir():
            raise ValueError("Output must be a valid directory for the CSV export")
        self.output_dir = output_dir
        self.parser: TrainingParser | None = None

    def open(self, parser=None) -> None:
        self.parser = parser

    def write_data(
        self, output: Path, entries: Sequence[TrainingEpoch | ValidationEpoch], dataclass: type
    ) -> None:
        if not entries:
            logger.warning(f"No {dataclass.__name__} entry, skipping.")
        with open(output, "w") as f:
            writer = csv.DictWriter(f, fieldnames=dataclass.__annotations__)
            writer.writeheader()
            for entry in entries:
                writer.writerow(vars(entry))

    def publish(self) -> None:
        assert self.parser is not None, "Parser must be set to run CSV publication."
        training_log = self.parser.output
        training_output = self.output_dir / "training.csv"
        if training_output.exists():
            logger.warning(f"Training output file {training_output} exists, skipping.")
        else:
            self.write_data(training_output, training_log.training, TrainingEpoch)

        validation_output = self.output_dir / "validation.csv"
        if validation_output.exists():
            logger.warning(f"Validation output file {validation_output} exists, skipping.")
        else:
            self.write_data(validation_output, training_log.validation, ValidationEpoch)


class WandB(Publisher):
    def __init__(
        self,
        *,
        project: str,
        group: str,
        name: str,
        suffix: str = "",
        # Optional path to a directory containing training artifacts
        artifacts: Path | None = None,
        artifacts_name: str = "logs",
        **extra_kwargs,
    ):
        from translations_parser.parser import TrainingParser

        # Set logging of wandb module to WARNING, so we output training logs instead
        self.wandb_logger = logging.getLogger("wandb")
        self.wandb_logger.setLevel(logging.ERROR)

        self.project = project
        self.group = group
        self.suffix = suffix
        # Build a unique run identifier based on the passed suffix
        # This ID is also used as display name on W&B, as the interface expects unique display names among runs
        self.run = f"{name}{suffix}"

        self.artifacts = artifacts
        self.artifacts_name = artifacts_name
        self.extra_kwargs = extra_kwargs
        self.parser: TrainingParser | None = None
        self.wandb: wandb.sdk.wandb_run.Run | wandb.sdk.lib.disabled.RunDisabled | None = None

    def close(self) -> None:
        if self.wandb is None:
            return

        # Publish artifacts
        if self.artifacts:
            artifact = wandb.Artifact(name=self.artifacts_name, type=self.artifacts_name)
            artifact.add_dir(local_path=str(self.artifacts.resolve()))
            self.wandb.log_artifact(artifact)

        if self.parser is not None:
            # Store Marian logs as the main log artifact, instead of W&B client runtime.
            # This will be overwritten in case an unhandled exception occurs.
            for line in self.parser.parsed_logs:
                sys.stdout.write(f"{line}\n")

        self.wandb.finish()

    def open(self, parser=None) -> None:
        self.parser = parser
        config = getattr(parser, "config", {}).copy()
        config.update(self.extra_kwargs.pop("config", {}))
        # Publish datasets stats directly in the dashboard
        datasets = config.pop("datasets", None)

        try:
            self.wandb = wandb.init(
                project=self.project,
                group=self.group,
                name=self.run,
                id=self.run,
                config=config,
                # Since we use unique run names based on group ID (e.g. finetune-student_MjcJG),
                # we can use "allow" mode for resuming a stopped Taskcluster run in case of preemption.
                # It will continue logging to the same run if it exists.
                # Offline publication should handle run deletion separately (use --override-runs).
                resume="allow",
                **self.extra_kwargs,
            )
            if self.wandb.resumed:
                logger.info(f"W&B run is being resumed from existing run '{self.run}'.")
        except Exception as e:
            logger.error(f"WandB client could not be initialized: {e}. No data will be published.")

        if datasets is not None:
            # Log dataset sizes as a custom bar chart
            self.wandb.log(
                {
                    "Datasets": wandb.plot.bar(
                        wandb.Table(
                            columns=["Name", "Count"],
                            data=[[key, value] for key, value in datasets.items()],
                        ),
                        "Name",
                        "Count",
                        title="Datasets",
                    )
                }
            )

    def generic_log(self, data: TrainingEpoch | ValidationEpoch) -> None:
        if self.wandb is None:
            return
        epoch = vars(data)
        step = epoch.pop("up")
        for key, val in epoch.items():
            if val is None:
                # Do not publish null values (e.g. perplexity in Marian 1.10)
                continue
            self.wandb.log(step=step, data={key: val})

    def handle_training(self, training: TrainingEpoch) -> None:
        self.generic_log(training)

    def handle_validation(self, validation: ValidationEpoch) -> None:
        self.generic_log(validation)

    def handle_metrics(self, metrics: Sequence[Metric]) -> None:
        if self.wandb is None:
            return
        for metric in metrics:
            title = metric.importer
            if metric.augmentation:
                title = f"{title}_{metric.augmentation}"
            if metric.dataset:
                title = f"{title}_{metric.dataset}"
            # Publish a bar chart (a table with values will also be available from W&B)
            self.wandb.log(
                {
                    title: wandb.plot.bar(
                        wandb.Table(
                            columns=["Metric", "Value"],
                            data=[
                                [key, getattr(metric, key)]
                                for key in METRIC_KEYS
                                if getattr(metric, key) is not None
                            ],
                        ),
                        "Metric",
                        "Value",
                        title=title,
                    )
                }
            )

    @classmethod
    def publish_group_logs(
        cls,
        *,
        logs_parent_folder: list[str],
        project: str,
        group: str,
        suffix: str,
        existing_runs: list[str] | None = None,
        snakemake: bool = False,
    ) -> None:
        """
        Publish files within `logs_dir` to W&B artifacts for a specific group.
        A fake W&B run named `group_logs` is created to publish those artifacts
        among with all evaluation files (quantized + experiments).
        If existing run is set, runs found not specified in this list will also
        be published to W&B.
        """
        from translations_parser.parser import TrainingParser

        try:
            if (
                len(
                    wandb.Api().runs(
                        path=project, filters={"display_name": "group_logs", "group": group}
                    )
                )
                > 0
            ):
                logger.warning("Skipping group_logs fake run publication as it already exists")
                return
        except ValueError as e:
            # Project may not exist yet as group_logs is published before the first training task
            if "could not find project" not in str(e).lower():
                logger.warning(f"Detection of a previous group_logs run failed: {e}")

        logs_dir = Path("/".join([*logs_parent_folder[:-1], "logs", project, group]))
        models_dir = Path("/".join([*logs_parent_folder[:-1], "models", project, group]))
        # Old experiments use `speed` directory for quantized metrics
        quantized_metrics = sorted(
            Path(
                "/".join(
                    [*logs_parent_folder[:-1], "models", project, group, "evaluation", "speed"]
                )
            ).glob("*.metrics")
        )
        logs_metrics = sorted((logs_dir / "eval").glob("eval*.log"))
        direct_metrics = sorted((logs_dir / "metrics").glob("*.metrics"))

        taskcluster_metrics = []
        # Do not retrieve metrics from models directory for legacy Snakemake experiments
        if snakemake is False:
            taskcluster_metrics = sorted((models_dir).glob("**/*.metrics"))

        if quantized_metrics:
            logger.info(f"Found {len(quantized_metrics)} quantized metrics from speed folder")
        if logs_metrics:
            logger.info(f"Found {len(logs_metrics)} metrics from task logs")
        if direct_metrics:
            logger.info(f"Found {len(direct_metrics)} Snakemake metrics from .metrics artifacts")
        if taskcluster_metrics:
            logger.info(
                f"Found {len(taskcluster_metrics)} Taskcluster metrics from .metrics artifacts"
            )

        # Store metrics by run name
        metrics = defaultdict(list)
        # Add metrics from the speed folder
        for file in quantized_metrics:
            importer, dataset = file.stem.split("_", 1)
            metrics["quantized"].append(Metric.from_file(file, importer=importer, dataset=dataset))
        # Add metrics from tasks logs
        for file in logs_metrics:
            try:
                model_name, importer, dataset, aug = parse_task_label(file.stem)
                with file.open("r") as f:
                    lines = f.readlines()
                metrics[model_name].append(
                    Metric.from_tc_context(
                        importer=importer, dataset=dataset, lines=lines, augmentation=aug
                    )
                )
            except ValueError as e:
                logger.error(f"Could not parse metrics from {file.resolve()}: {e}")

        # Add metrics from old SnakeMake .metrics files
        for file in direct_metrics:
            model_name, importer, dataset, aug = parse_task_label(file.stem)
            try:
                metrics[model_name].append(
                    Metric.from_file(file, importer=importer, dataset=dataset, augmentation=aug)
                )
            except ValueError as e:
                logger.error(f"Could not parse metrics from {file.resolve()}: {e}")

        # Add metrics from new Taskcluster .metrics files
        for file in taskcluster_metrics:
            model_name = patch_model_name(file.parent.name)
            try:
                metric_attrs = parse_gcp_metric(file.stem)
                metrics[model_name].append(
                    Metric.from_file(
                        file,
                        importer=metric_attrs.importer,
                        dataset=metric_attrs.dataset,
                        augmentation=metric_attrs.augmentation,
                    )
                )
            except ValueError as e:
                logger.error(f"Could not parse metrics from {file.resolve()}: {e}")

        # Publish missing runs (runs without training data)
        missing_run_metrics = {}
        if existing_runs is not None:
            missing_run_metrics = {
                name: metrics for name, metrics in metrics.items() if name not in existing_runs
            }

        for model_name, model_metrics in missing_run_metrics.items():
            logger.info(f"Creating missing run {model_name} with associated metrics")
            publisher = cls(
                project=project,
                group=group,
                name=model_name,
                suffix=suffix,
            )
            publisher.open(TrainingParser(logs_iter=iter([]), publishers=[]))
            publisher.handle_metrics(model_metrics)
            publisher.close()

        # Publication of the `group_logs` fake run
        config = {}
        config_path = Path(
            "/".join([*logs_parent_folder[:-1], "experiments", project, group, "config.yml"])
        )
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

        publisher = cls(
            project=project,
            group=group,
            name="group_logs",
            suffix=suffix,
        )
        publisher.wandb = wandb.init(
            project=project,
            group=group,
            name=publisher.run,
            id=publisher.run,
            config=config,
        )

        if metrics:
            # Publish all evaluation metrics to a table
            table = wandb.Table(
                columns=["Group", "Model", "Importer", "Dataset", "Augmenation", *METRIC_KEYS],
                data=[
                    [group, run_name, metric.importer, metric.dataset, metric.augmentation]
                    + [getattr(metric, attr) for attr in METRIC_KEYS]
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
