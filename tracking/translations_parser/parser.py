import logging
import os
import re
import shlex
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from itertools import tee
from pathlib import Path
from typing import Callable, DefaultDict, List

import yaml

from translations_parser.data import Metric, TrainingEpoch, TrainingLog, ValidationEpoch
from translations_parser.publishers import Publisher
from translations_parser.utils import get_lines_count

logger = logging.getLogger(__name__)

HEADER_RE = re.compile(r"(?<=\[)(?P<value>.+?)\] ")
#                            \[               \]   Match square brackets
#                        (?<=\[)                   This is a positive lookbehind, that looks for the first "[" in a line.
#                                                  This line does not actually capture
#                               (?P<value>.+?)     Non-greedily match the "value" inside of the square brackes.
VALIDATION_RE = re.compile(
    r"Ep\.[ :]+(?P<ep>\d+)"
    r"[ :]+Up\.[ :]+(?P<up>\d+)"
    r"[ :]+(?P<key>[\w-]+)"
    r"[ :]+(?P<value>[\d\.]+)"
    r"([ :]+stalled (?P<stalled>\d+) times)?"
)
TRAINING_RE = re.compile(
    r"Ep\.[ :]+(?P<epoch>\d+)[ :]+"
    r"Up\.[ :]+(?P<up>\d+)[ :]+"
    r"Sen\.[ :]+(?P<sen>[\d,]+)[ :]+"
    r"Cost[ :]+(?P<cost>[\d.]+)[ :]+"
    r"Time[ :]+(?P<time>[\d\.]+)s[ :]+"
    r"(?P<rate>[\d\.]+) words\/s[ :]+"
    r"gNorm[ :]+(?P<gnorm>[\d\.]+)"
    r"([ :]+L.r. (?P<learning_rate>[\d\.e-]+))?"
)

# Expected version of Marian for a clean parsing
SUPPORTED_MARIAN_VERSIONS = [(1, 10), (1, 12)]

MARIAN_ARGS_REGEX = re.compile(r"command line:[\n ]+[\w\/-]+\/marian +(.*)")
# Last Marian command line argument (not being part of training extra arguments)
LAST_MARIAN_DECLARED_ARGUMENT = "seed"


class TrainingParser:
    def __init__(
        self,
        logs_iter: Iterable[str],
        publishers: Sequence[Publisher],
        log_filter: Callable | None = None,
        skip_marian_context: bool = False,
        metrics: Sequence[Metric] | None = None,
    ) -> None:
        # Iterable reading logs lines
        self.logs_iter = logs_iter
        # Function to exclude log lines depending on the headers
        self.log_filter = log_filter
        self._current_index = 0
        self.parsed = False
        self.config: dict = {}
        self.parsed_logs: List[str] = []
        # Optional list of Metric published earlier to the parsing
        self.metrics = metrics
        self.training: list[TrainingEpoch] = []
        self.validation: list[ValidationEpoch] = []
        # Dict mapping (epoch, up) to values parsed on multiple lines
        self._validation_entries: DefaultDict[tuple[int, int], dict] = defaultdict(dict)
        # Option to read logs directly (skip check for Marian context)
        self.skip_marian_context = skip_marian_context
        # Marian exection data
        self.version: str | None = None
        self.version_hash: str | None = None
        self.release_date: str | None = None
        self.run_date: datetime | None = None
        self.description: str | None = None
        # Data publication after parsing logs
        self.publishers = publishers

    def get_headers(self, line: str) -> tuple[list[tuple[str]], int]:
        """
        Returns a list of tuples representing all headers of a log line
        and the position of the last character representing the header.
        """
        matches = list(HEADER_RE.finditer(line))
        if not matches:
            return ([], 0)
        return ([tuple(m.group("value").split()) for m in matches], matches[-1].span()[-1])

    def get_timestamp(self, headers: Sequence[Sequence[str]]) -> datetime | None:
        """
        Looks for a timestamp in Taskcluster header tags.
        Returns None in case no timestamp is found.
        """
        for values in headers:
            if len(values) != 2:
                continue
            base, timestamp = values
            # TC adds a timestamp after the task header
            if base == "task":
                try:
                    return datetime.fromisoformat(timestamp.rstrip("Z"))
                except ValueError as e:
                    # Taskcluster timestamp should always be a valid date
                    logger.error(
                        f"Unreadable taskcluster timestamp line {self._current_index}: {e}"
                    )
            # Marian timestamp is composed of two values, one for date and one for hour precision
            try:
                return datetime.fromisoformat("T".join(values))
            except ValueError:
                continue
        return None

    def parse_training_log(self, text: str) -> TrainingEpoch | None:
        match = TRAINING_RE.match(text)
        if not match:
            return None
        # Filter out null values
        values = {k: v for k, v in match.groupdict().items() if v is not None}
        # Update sen from 1,234,567 to 1_234_567 that Python can interpret
        values["sen"] = values["sen"].replace(",", "_")
        # Cast values to match output types
        casted_values = {
            k: (
                TrainingEpoch.__annotations__[k](v)
                if callable(TrainingEpoch.__annotations__[k])
                else float(v)
            )
            for k, v in values.items()
        }
        training_epoch = TrainingEpoch(**casted_values)
        self.training.append(training_epoch)
        for publisher in self.publishers:
            try:
                publisher.handle_training(training_epoch)
            except Exception as e:
                logger.error(
                    f"Error publishing training epoch using {publisher.__class__.__name__}: {e}"
                )
        return training_epoch

    def parse_validation_log(
        self, headers: Sequence[Sequence[str]], text: str
    ) -> ValidationEpoch | None:
        """Parses a validation entry on multiple lines."""
        if ("valid",) not in headers or not (match := VALIDATION_RE.match(text)):
            return None
        results = match.groupdict()
        # Replace items keys to match ValidationEpoch dataclass
        key = results["key"].replace("-", "_")
        epoch, up = int(results["ep"]), int(results["up"])
        entry = self._validation_entries[(epoch, up)]
        # Transform values to match output types
        entry[key] = ValidationEpoch.__annotations__[key](results["value"])
        if results["stalled"] is not None:
            entry[f"{key}_stalled"] = float(results["stalled"])
        # Build a validation epoch from multiple lines
        expected_keys = set(
            key
            for key in ValidationEpoch.__annotations__.keys()
            if not (
                # Stalled data are not necessary present on validation entries
                key.endswith("_stalled")
                or key in ("epoch", "up", "perplexity")
            )
        )
        if not (expected_keys - set(entry.keys())):
            validation_epoch = ValidationEpoch(epoch=epoch, up=up, **entry)
            self.validation.append(validation_epoch)
            for publisher in self.publishers:
                try:
                    publisher.handle_validation(validation_epoch)
                except Exception as e:
                    logger.error(
                        f"Error publishing validation epoch using {publisher.__class__.__name__}: {e}"
                    )
            del self._validation_entries[(epoch, up)]
            return validation_epoch
        return None

    def _iter_log_entries(self) -> Iterator[tuple[list[tuple[str]], str]]:
        """
        Inner method to iterate on log lines passed to
        the parser, differentiating headers and text.
        Automatically set Marian run date when found.
        """
        for line in self.logs_iter:
            # When reading stdin stream, propagate raw lines to stdout
            # and force flush on stdout to make sure every line gets displayed
            sys.stdout.buffer.write(line.encode("utf-8"))
            sys.stdout.buffer.flush()

            self._current_index += 1
            headers, position = self.get_headers(line)
            if self.log_filter and not self.log_filter(headers):
                logger.debug(
                    f"Skipping line {self._current_index} : Headers does not match the filter"
                )
                continue
            elif self.run_date is None:
                # Try to fill run date from log headers
                self.run_date = self.get_timestamp(headers)
            text = line[position:]

            def _join(seq):
                if not seq:
                    return None
                if isinstance(seq[0], str):
                    return "_".join([item for item in seq if item is not None])
                return _join([_join(item) for item in seq if item is not None])

            # Record logs depending on Marian headers
            tag = None
            if len(headers) >= 2:
                # The 2 first headers are ignored (task timestamp, then marian timestamp)
                _, _, *marian_tags = headers
                tag = _join(marian_tags)
            if tag:
                self.parsed_logs.append(f"[tag] {text}")
            else:
                self.parsed_logs.append(text)

            yield headers, text

    def get_extra_marian_config(self) -> dict:
        """
        Read extra configuration files (Marian, OpusTrainer, extra CLI arguments).
        Publication outside of a Taskcluster context (offline mode) cannot access
        the configuration files, only extra-args will be set in this case.
        """
        extra_config = {
            "arguments": None,
            "model": None,
            "training": None,
            "datasets": None,
            "opustrainer": None,
        }

        if (
            self.description is None
            or (match := MARIAN_ARGS_REGEX.search(self.description)) is None
        ):
            logger.error(self.description)
            logger.warning(
                "Invalid Marian description, skipping Marian and OpusTrainer configuration detection."
            )
            return extra_config

        logger.info("Reading Marian command line arguments.")
        (arguments_str,) = match.groups()
        # Build args from the command line input text
        args = defaultdict(list)
        key = None
        for i in iter(shlex.split(arguments_str)):
            if i.startswith("-"):
                key = i.strip("-")
                continue
            args[key].append(i)

        # Store arguments used to run Marian, flattening single values
        def flatten(vals):
            if not vals:
                return ""
            elif len(vals) == 1:
                return vals[0]
            return vals

        extra_config["arguments"] = {k: flatten(v) for k, v in args.items()}

        if os.environ.get("TASK_ID") is None:
            logger.info(
                "Extra configuration files can only be retrieved in Taskcluster context, skipping."
            )
            return extra_config

        # Handle Marian model and training YAML configuration files (called as --config or -c)
        for path in args.get("config", args["c"]):
            if path.startswith("configs/training"):
                key = "training"
            elif path.startswith("configs/model"):
                key = "model"
            else:
                continue
            try:
                with open(path, "r") as f:
                    extra_config[key] = yaml.safe_load(f.read())
            except Exception as e:
                logger.warning(f"Impossible to parse Marian {key} config at {path}: {e}")

        # Handle OpusTrainer configuration
        (model_path,) = args.get("model", ("./model.npz",))
        model_dir = Path(model_path).parent
        train_conf_path = (model_dir / "config.opustrainer.yml").resolve()
        if not train_conf_path.exists():
            logger.warning(f"OpusTrainer configuration file does not exists at {train_conf_path}.")
        else:
            try:
                with open(train_conf_path, "r") as f:
                    extra_config["opustrainer"] = yaml.safe_load(f.read())
            except Exception as e:
                logger.warning(f"Impossible to parse OpusTrainer config at {train_conf_path}: {e}")
            else:
                logger.info("Reading datasets statistics from OpusTrainer configuration.")
                try:
                    dataset_conf = extra_config.get("opustrainer", {}).get("datasets", {})
                    extra_config["datasets"] = {
                        key: get_lines_count(path) for key, path in dataset_conf.items()
                    }
                except Exception as e:
                    logger.warning(
                        f"OpusTrainer configuration could not be read at {train_conf_path}: {e}."
                    )

        return extra_config

    def parse_marian_context(self, logs_iter: Iterator[tuple[list[tuple[str]], str]]) -> None:
        """
        Looks for Marian context in the first logs lines.
        Returns the first headers and text couple that is not Marian context.
        """
        headers: list[tuple[str]] = []
        # Consume first lines until we get the Marian header
        while ("marian",) not in headers:
            try:
                headers, text = next(logs_iter)
                logger.debug(f"Marian header not found in: headers={headers} text={text.strip()}")
            except StopIteration:
                raise ValueError("Could not find a [marian] entry in the training log.")

        logger.debug(f"Reading Marian version from text={text.strip()}")
        _, version, self.version_hash, self.release_date, *_ = text.split()
        version = version.rstrip(";")
        major, minor = map(int, version.lstrip("v").split(".")[:2])
        self.version = f"{major}.{minor}"
        logger.info(f"Detected Marian version {self.version}")
        if (major, minor) not in SUPPORTED_MARIAN_VERSIONS:
            versions = ", ".join(f"{major}.{minor}" for major, minor in SUPPORTED_MARIAN_VERSIONS)
            logger.warning(
                f"Parsing logs from a non supported Marian version {major}.{minor} "
                f"(supported versions: {versions})."
            )

        logger.debug("Reading Marian run description.")
        desc = []
        for headers, text in logs_iter:
            # Marian headers stops when dumping the configuration
            if ("config",) in headers:
                break
            desc.append(text)
        self.description = " ".join(desc)

        # Try to parse all following config lines as YAML
        logger.debug("Reading Marian configuration.")
        config_yaml = ""
        while ("config",) in headers:
            # Marian incorrectly logs some messages with [config] prefix.
            if "Model is being created" in text or "Loaded model has been created" in text:
                headers, text = next(logs_iter)
                break
            config_yaml += f"{text}\n"
            headers, text = next(logs_iter)
        try:
            self.config["marian"] = yaml.safe_load(config_yaml)
        except Exception as e:
            logger.error(f"Impossible to parse Marian config YAML: {e}")

        # Try to read required extra configuration files when running online from Taskcluster
        self.config.update(self.get_extra_marian_config())

    def parse_data(self, logs_iter: Iterator[tuple[list[tuple[str]], str]]) -> None:
        """
        Iterates logs until the end to find training or validation
        data and report incomplete multiline logs.
        """
        while True:
            try:
                headers, text = next(logs_iter)
                training = self.parse_training_log(text)
                if not training:
                    self.parse_validation_log(headers, text)
            except StopIteration:
                break
        if self._validation_entries.keys():
            logger.warning(
                "Some validation data is incomplete with the following epoch/up couples:"
            )
            for epoch, up in self._validation_entries.keys():
                logger.warning(f"* Ep. {epoch}, Up. {up}")

    def parse(self) -> None:
        """
        Parses and publishes training logs:
          1. Optionally reads Marian context (version, configuration)
          2. Looks for training or validation data among next lines
        """
        if self.parsed:
            raise Exception("The parser already ran.")
        logs_iter = self._iter_log_entries()

        logger.info("Reading logs stream.")
        if not self.skip_marian_context:
            # Copy logs iterable so we avoid reading out of context lines.
            # This will not affect inner self._current_index, as we stop incrementing after reading the context.
            logs_iter, copy = tee(logs_iter)
            self.parse_marian_context(copy)

        for publisher in self.publishers:
            publisher.open(self)

        # Run training and validation data parser
        self.parse_data(logs_iter)
        self.parsed = True

        # Once all data has been parsed, call the final publication API
        for publisher in self.publishers:
            try:
                publisher.publish()
                # Publish optional metrics
                if self.metrics:
                    publisher.handle_metrics(self.metrics)
            except Exception as e:
                logger.error(f"Error publishing data using {publisher.__class__.__name__}: {e}")

        for publisher in self.publishers:
            publisher.close()

    @property
    def output(self) -> TrainingLog:
        if not self.parsed:
            raise Exception("Please run the parser before reading the output")
        return TrainingLog(
            run_date=self.run_date,
            configuration=self.config,
            training=self.training,
            validation=list(self.validation),
            logs=self.parsed_logs,
        )

    def run(self) -> None:
        """Parse logs stream."""
        try:
            self.parse()
        except StopIteration:
            raise ValueError("Not all required lines were found from the log file.")

        count = len(self.parsed_logs)
        logger.info(f"Successfully parsed {count} lines")
        logger.info(f"Found {len(self.training)} training entries")
        logger.info(f"Found {len(self.validation)} validation entries")
