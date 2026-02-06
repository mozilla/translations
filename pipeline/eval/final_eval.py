"""
Run final evaluation for exported models and compare to other translators

To run locally:

task inference-build
pip install -r taskcluster/docker/eval/final_eval.txt
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$(pwd)
python pipeline/eval/final_eval.py
    --config=taskcluster/configs/eval.ci.yml
    --artifacts=data/final_evals
    --bergamot-cli=inference/build/src/app/translator-cli

"""
import argparse
import gc
import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Any

import requests
import yaml
from toolz import groupby

from pipeline.common.downloads import location_exists
from pipeline.common.logging import get_logger
from pipeline.eval.eval_datasets import Flores200Plus, Wmt24pp, Bouquet
from pipeline.eval.metrics import (
    Chrfpp,
    Bleu,
    Chrf,
    Comet22,
    MetricX24,
    Metricx24Qe,
    MetricResults,
    LlmRef,
    RegularMetric,
    SpBleu,
    SubprocessMetric,
)
from pipeline.eval.translators import (
    BergamotTranslator,
    OpusmtTranslator,
    GoogleTranslator,
    MicrosoftTranslator,
    NllbTranslator,
    BergamotPivotTranslator,
)
from pipeline.langs.codes import LanguageNotSupported

logger = get_logger(__file__)
logger.setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("argostranslate").disabled = True
logging.getLogger("argostranslate.utils").disabled = True

PIVOT_PAIRS = {("de", "fr"), ("fr", "de"), ("it", "de")}
ALL_METRICS = [Chrf, Chrfpp, Bleu, SpBleu, Comet22, MetricX24, Metricx24Qe, LlmRef]
METRICX_VENV = Path("/tmp/metricx-venv")
METRICX_REQUIREMENTS = Path(__file__).parent / "requirements" / "metricx.txt"
METRICX_CLASSES = {MetricX24, Metricx24Qe}
ALL_DATASETS = {d.name: d for d in [Flores200Plus, Wmt24pp, Bouquet]}
ALL_TRANSLATORS = [
    BergamotTranslator,
    OpusmtTranslator,
    GoogleTranslator,
    MicrosoftTranslator,
    # https://github.com/mozilla/translations/issues/1309
    # ArgosTranslator,
    NllbTranslator,
]
PROD_BUCKET = "moz-fx-translations-data--303e-prod-translations-data"


def with_retry(fn, retries=5, initial_delay=60, description="operation"):
    """Retry a function with exponential backoff for HuggingFace rate limiting."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            delay = initial_delay * (2**attempt)
            logger.warning(
                f"{description} failed (attempt {attempt + 1}/{retries}): {e}. "
                f"Retrying in {delay}s..."
            )
            time.sleep(delay)


class Config:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
            description=__doc__,
            # Preserves whitespace in the help text.
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument(
            "--config",
            required=False,
            type=Path,
            help="The path to the yaml config file. If present, ignores other arguments",
        )
        parser.add_argument(
            "--artifacts", required=True, type=Path, help="The path to the artifacts folder"
        )
        parser.add_argument(
            "--bergamot-cli",
            required=True,
            type=Path,
            help="The path to the Bergamot Translator CLI app",
        )
        parser.add_argument(
            "--override",
            required=False,
            type=bool,
            help="Whether to rerun evals even if they already exist in the storage.",
        )
        parser.add_argument(
            "--ignore-fails",
            required=False,
            type=bool,
            help="Whether to continue running if a metric fails for some datasets. "
            "Useful for intermittent failures of non-deterministic metrics like LLM-based ones.",
        )
        parser.add_argument(
            "--storage",
            required=False,
            default="gcs",
            type=str,
            choices=["local", "gcs"],
            help="Storage type to check if results are already present: local or GCS",
        )
        parser.add_argument(
            "--bucket",
            required=False,
            default=PROD_BUCKET,
            type=str,
            help="GCS bucket to check for results when using `--storage gcs`",
        )
        parser.add_argument(
            "--languages",
            required=False,
            nargs="+",
            type=str,
            help="Language pairs to evaluate, e.g., en-ru de-it",
        )
        parser.add_argument(
            "--datasets",
            required=False,
            nargs="+",
            type=str,
            choices=ALL_DATASETS.keys(),
            help="Evaluation datasets",
        )
        parser.add_argument(
            "--translators",
            required=False,
            nargs="+",
            type=str,
            choices=[t.name for t in ALL_TRANSLATORS],
            help="Translation systems to run",
        )
        parser.add_argument(
            "--metrics",
            required=False,
            nargs="+",
            type=str,
            choices=[m.name for m in ALL_METRICS],
            help="Evaluation metrics",
        )
        parser.add_argument(
            "--models",
            required=False,
            nargs="+",
            type=str,
            help="Bergamot models to run",
        )

        args = parser.parse_args()
        self.artifacts_path = str(args.artifacts)
        self.bergamot_cli_path = str(args.bergamot_cli)
        if args.config:
            if logger.level == logging.DEBUG:
                with open(args.config, "r") as f:
                    logger.debug("Config text: ")
                    logger.debug(f.read())
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            # to test with taskcluster config
            evals_config = config["evals"] if "evals" in config else config
            self.override = evals_config["override"]
            self.ignore_fails = evals_config["ignore-fails"]
            self.storage = evals_config["storage"]
            self.bucket = evals_config["bucket"]
            self.languages = evals_config["languages"]
            self.datasets = evals_config["datasets"]
            self.translators = evals_config["translators"]
            self.metrics = evals_config["metrics"]
            self.models = evals_config["models"]
            self._validate()
        else:
            self.override = args.override
            self.ignore_fails = args.ignore_fails
            self.storage = args.storage
            self.bucket = args.bucket
            self.languages = args.languages
            self.datasets = args.datasets
            self.translators = args.translators
            self.metrics = args.metrics
            self.models = args.models

    def print(self) -> str:
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
        )

    def _validate(self):
        if self.storage and self.storage not in ["local", "gcs"]:
            raise ValueError(f"Invalid storage: {self.storage}. Must be 'local' or 'gcs'")

        valid_datasets = ALL_DATASETS.keys()
        if self.datasets:
            for dataset in self.datasets:
                if dataset not in valid_datasets:
                    raise ValueError(
                        f"Invalid dataset: {dataset}. Must be one of {valid_datasets}"
                    )

        valid_translators = [t.name for t in ALL_TRANSLATORS]
        if self.translators:
            for translator in self.translators:
                if translator not in valid_translators:
                    raise ValueError(
                        f"Invalid translator: {translator}. Must be one of {valid_translators}"
                    )

        valid_metrics = [m.name for m in ALL_METRICS]
        if self.metrics:
            for metric in self.metrics:
                if metric not in valid_metrics:
                    raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")


@dataclass
class EvalsMeta:
    src: str
    trg: str
    dataset: str
    translator: str
    model_name: str

    def format_path(self) -> Path:
        return Path(f"{self.src}-{self.trg}/{self.dataset}/{self.translator}/{self.model_name}")


@dataclass()
class Translation:
    src_text: str | None
    trg_text: str
    ref_text: str | None

    def to_dict(self) -> dict:
        return {"src": self.src_text, "trg": self.trg_text, "ref": self.ref_text}

    @staticmethod
    def from_dict(d: dict):
        return Translation(d["src"], d["trg"], d["ref"])


class Storage:
    LATEST = "latest"
    METRICS = "metrics.json"
    TRANSLATIONS = "translations.json"
    SCORES = "scores.json"
    # file name parts separator (different from "/" due to Taskcluster directory uploading limitation)
    SEPARATOR = "__"

    def __init__(self, write_path: Path, read_bucket: str | None = None):
        self.write_path = write_path
        self.read_path = (
            f"https://storage.googleapis.com/{read_bucket}/final-evals"
            if read_bucket
            else str(write_path)
        )
        self._gcs_items = self._list_gcs(read_bucket) if read_bucket else None

    def translation_exists(self, meta: EvalsMeta):
        return self._file_exists(meta.format_path() / self.LATEST / self.TRANSLATIONS)

    def metric_exists(self, meta: EvalsMeta, name: str):
        return self._file_exists(
            meta.format_path() / self.LATEST / f"{name}.{self.METRICS}"
        ) and self._file_exists(meta.format_path() / self.LATEST / f"{name}.{self.SCORES}")

    def save_translations(
        self, meta: EvalsMeta, timestamp: str, translations: list[Translation]
    ) -> Path:
        timestamp_path = meta.format_path() / timestamp
        json_obj = [tr.to_dict() for tr in translations]
        self._write(json_obj, timestamp_path / self.TRANSLATIONS)
        self._write(json_obj, meta.format_path() / self.LATEST / self.TRANSLATIONS)

        return timestamp_path

    def load_translations(self, meta: EvalsMeta) -> list[Translation]:
        tr_json = self._load(meta.format_path() / self.LATEST / self.TRANSLATIONS)
        return [Translation.from_dict(tr) for tr in tr_json]

    def save_metric(self, meta: EvalsMeta, timestamp: str, metric: MetricResults) -> Path:
        timestamp_path = meta.format_path() / timestamp

        metrics_json = {"score": round(metric.corpus_score, 4), "details": metric.details}
        # score is a single number for most metrics but there are exceptions like LLM-produced multiple scores with commentary
        scores_json = [round(s, 4) if isinstance(s, float) else s for s in metric.segment_scores]

        self._write(metrics_json, timestamp_path / f"{metric.name}.{self.METRICS}")
        self._write(scores_json, timestamp_path / f"{metric.name}.{self.SCORES}")
        self._write(
            metrics_json, meta.format_path() / self.LATEST / f"{metric.name}.{self.METRICS}"
        )
        self._write(scores_json, meta.format_path() / self.LATEST / f"{metric.name}.{self.SCORES}")

        return timestamp_path

    @staticmethod
    def _list_gcs(bucket: str) -> set[str]:
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
        items = set()
        page_token = None
        logger.info(f"Listing evals on Google Cloud Storage bucket {bucket}...")

        while True:
            params = {"prefix": "final-evals"}
            if page_token:
                params["pageToken"] = page_token

            response = requests.get(url, params=params).json()
            items.update(
                item["name"].replace("final-evals/", "") for item in response.get("items", [])
            )

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return items

    def _file_exists(self, path: Path) -> bool:
        file_path = self._get_file_name(path)

        if self._gcs_items and file_path in self._gcs_items:
            return True

        return location_exists(f"{self.read_path}/{file_path}")

    def _write(self, data: object, path: Path):
        full_path = self.write_path / self._get_file_name(path)
        os.makedirs(full_path.parent, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self, path: Path) -> Any:
        write_path = self.write_path / self._get_file_name(path)
        if write_path.exists():
            # load recently saved translations
            with open(write_path, "r", encoding="utf-8") as f:
                tr_json = json.load(f)
        else:
            # load translations saved by previous runs
            read_path = f"{self.read_path}/{self._get_file_name(path)}"
            tr_json = requests.get(str(read_path)).json()

        return tr_json

    def _get_file_name(self, name: Path) -> str:
        return str(name).replace("/", self.SEPARATOR)


class Secrets:
    def __init__(self):
        import taskcluster

        root_url = os.environ.get("TASKCLUSTER_PROXY_URL")
        assert root_url, "When running in Taskcluster the TASKCLUSTER_PROXY_URL must be set."
        self.secrets = taskcluster.Secrets({"rootUrl": root_url})

    def prepare_keys(self):
        logger.info("Reading secrets from Taskcluster")
        os.environ["HF_TOKEN"] = self.read_key("huggingface")["token"]
        os.environ["OPENAI_API_KEY"] = self.read_key("chatgpt")["token"]
        os.environ["AZURE_TRANSLATOR_KEY"] = self.read_key("azure-translate")["token"]
        google_key_file = tempfile.NamedTemporaryFile("w", delete=False)
        with google_key_file:
            json.dump(self.read_key("google-translate"), google_key_file)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_key_file.name

    def read_key(self, name: str) -> dict:
        try:
            response = self.secrets.get(f"project/translations/level-1/{name}")
            return response["secret"]
        except Exception as e:
            raise ValueError(f"Could not retrieve the secret key {name}: {e}")


class EvalsRunner:
    def __init__(self, config: Config):
        logger.info(f"Config: {config.print()}")
        self.config = config
        self.storage = Storage(
            write_path=Path(config.artifacts_path),
            read_bucket=config.bucket if config.storage == "gcs" else None,
        )
        self.run_timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        # Check if we're inside a Taskcluster task
        if os.environ.get("TASK_ID"):
            secrets = Secrets()
            secrets.prepare_keys()

        self.translators_cls = [
            t for t in ALL_TRANSLATORS if not config.translators or t.name in config.translators
        ]
        self.metrics_cls = [
            m for m in ALL_METRICS if not config.metrics or m.name in config.metrics
        ]
        self.datasets_cls = [
            d for d in ALL_DATASETS.values() if not config.datasets or d.name in config.datasets
        ]

    def run(self):
        if self.config.languages:
            lang_pairs = [lp.split("-") for lp in self.config.languages]
        else:
            logger.info("Listing all Bergamnot models")
            bergamot_models = BergamotTranslator.list_all_models(self.config.bucket)
            lang_pairs = {(m.src, m.trg) for m in bergamot_models}.union(PIVOT_PAIRS)

        metrics_to_run = defaultdict(list)
        for src, trg in lang_pairs:
            logger.info(f"Processing {src} -> {trg}")
            for metric, metas in self.translate(src, trg).items():
                metrics_to_run[metric].extend(metas)

        if metrics_to_run:
            logger.info(f"Running {len(metrics_to_run)} metrics for translations")
            self.run_metrics(metrics_to_run)
        else:
            logger.info("No metrics to run")

    def translate(self, src: str, trg: str) -> dict[type[RegularMetric], list[EvalsMeta]]:
        metrics_to_run = defaultdict(list[EvalsMeta])

        for dataset_cls in self.datasets_cls:
            try:
                dataset = dataset_cls(src, trg)
                logger.debug(f"Running for dataset {dataset_cls.name}")
            except LanguageNotSupported:
                logger.debug(
                    f"Skipping dataset {dataset_cls.name}, it does not support {src} -> {trg}"
                )
                continue

            for translator_cls in self.translators_cls:
                model_names = None
                translator = None
                if translator_cls is BergamotTranslator:
                    if src != "en" and trg != "en" and translator_cls is BergamotTranslator:
                        translator_cls_new = BergamotPivotTranslator
                    else:
                        translator_cls_new = BergamotTranslator
                    translator = translator_cls_new(
                        src, trg, self.config.bucket, self.config.bergamot_cli_path
                    )

                    if self.config.models:
                        if len(self.config.models) == 1 and self.config.models[0] == "latest":
                            model_names = translator.list_latest_models()
                            logger.debug(f"Using latest Bergamot model only {model_names}")
                        else:
                            model_names = [
                                m for m in translator.list_models() if m in self.config.models
                            ]
                            logger.debug(f"Using specified Bergamot models {model_names}")
                else:
                    try:
                        translator = translator_cls(src, trg)
                    except LanguageNotSupported:
                        logger.warning(
                            f"Language pair {src}-{trg} is not supported by translator {translator_cls.name}"
                        )
                        continue

                if not model_names:
                    model_names = translator.list_models()
                    logger.debug(f"Using models {model_names}")

                for model_name in model_names:
                    meta = EvalsMeta(
                        src=src,
                        trg=trg,
                        dataset=dataset_cls.name,
                        translator=translator.name,
                        model_name=model_name,
                    )

                    if not self._needs_translation(meta, metrics_to_run):
                        logger.debug("Skipping translation")
                        continue

                    ref_texts, source_texts = self._load_texts(dataset)

                    logger.info(
                        f"Running translator {translator.name}, model {model_name}, dataset {dataset_cls.name}"
                    )
                    with_retry(
                        lambda t=translator, m=model_name: t.prepare(m),
                        description=f"translator.prepare({model_name})",
                    )
                    logger.info(f"Translating {len(source_texts)} texts")
                    translations = translator.translate(source_texts)

                    self._save_translations(dataset, meta, ref_texts, source_texts, translations)

                del translator
                self._clean()

        return metrics_to_run

    def _needs_translation(self, meta: EvalsMeta, metrics_to_run) -> bool:
        # Translate only if the metric supports the language pair and doesn't exist in the storage
        needs_metrics = False
        for metric in self.metrics_cls:
            if not metric.supports_lang(meta.src, meta.trg):
                logger.debug(f"Metric {metric.name} does not support {meta.src} -> {meta.trg}")
            elif not self.config.override and self.storage.metric_exists(meta, metric.name):
                logger.debug(f"Metric {metric.name} already exists for {meta.format_path()}")
            else:
                metrics_to_run[metric].append(meta)
                needs_metrics = True

        needs_translation = self.config.override or not self.storage.translation_exists(meta)
        if not needs_translation:
            logger.debug(f"Translations already exist for {meta.format_path()}")

        return needs_metrics and needs_translation

    @staticmethod
    def _load_texts(dataset):
        logger.info(f"Downloading dataset {dataset.name}")
        with_retry(dataset.download, description=f"dataset.download({dataset.name})")
        segments = dataset.get_texts()
        source_texts = [s.source_text for s in segments]
        ref_texts = [s.ref_text for s in segments]
        return ref_texts, source_texts

    def _save_translations(self, dataset, meta, ref_texts, source_texts, translations):
        # Do not save source and target sentences for restricted datasets
        if not dataset.is_restricted:
            to_save = [
                Translation(s, t, r) for s, t, r in zip(source_texts, translations, ref_texts)
            ]
        else:
            to_save = [Translation(None, tr, None) for tr in translations]
        saved_path = self.storage.save_translations(meta, self.run_timestamp, to_save)
        logger.info(f"Translations saved to {saved_path}")

    def run_metrics(self, to_run: dict[type[RegularMetric], list[EvalsMeta]]):
        # Group by metric to load GPU metrics once
        for metric_cls, all_metas in to_run.items():
            logger.info(f"Running metric {metric_cls.name}")
            if metric_cls in METRICX_CLASSES:
                metric = with_retry(
                    lambda cls=metric_cls: SubprocessMetric(
                        cls, METRICX_VENV, METRICX_REQUIREMENTS
                    ),
                    description=f"metric construction ({metric_cls.name})",
                )
            else:
                metric = with_retry(
                    metric_cls, description=f"metric construction ({metric_cls.name})"
                )
            # Then group by a language pair
            for pair, pair_metas in groupby(lambda m: (m.src, m.trg), all_metas).items():
                src, trg = pair
                # Then group by a dataset to load it once
                for dataset_name, dataset_metas in groupby(
                    lambda m: m.dataset, pair_metas
                ).items():
                    dataset = ALL_DATASETS[dataset_name](src, trg)
                    ref_texts, source_texts = self._load_texts(dataset)

                    for meta in dataset_metas:
                        translations = [tr.trg_text for tr in self.storage.load_translations(meta)]

                        logger.info(
                            f"Scoring {len(ref_texts)} texts with {metric.name} for dataset {dataset_name}, translator {meta.translator}, model {meta.model_name}, language pair {src}-{trg}"
                        )
                        try:
                            metric_results = metric.score(
                                meta.src, meta.trg, source_texts, translations, ref_texts
                            )
                            saved_path = self.storage.save_metric(
                                meta, self.run_timestamp, metric_results
                            )
                            logger.info(f"Metric saved to {saved_path}")
                        except Exception as e:
                            logger.error(
                                f"Failed to score metric {metric.name} for dataset {dataset_name}, translator {meta.translator}, model {meta.model_name}, language pair {src}-{trg}",
                                exc_info=e,
                            )
                            if not self.config.ignore_fails:
                                raise e

            del metric
            self._clean()

    @staticmethod
    def _clean():
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    EvalsRunner(Config()).run()
