"""
Run final evaluation for exported models and compare to other translators

To run locally:

task inference-build
pip install -r taskcluster/docker/eval/final_eval.txt
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH=$(pwd)
python pipeline/eval/final_eval.py
    --config=taskcluster/configs/eval.yml
    --artifacts=data/final_evals
    --bergamot-cli=inference/build/src/app/translator-cli

"""
import argparse
import gc
import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Any

import requests
import yaml

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
)
from pipeline.eval.translators import (
    BergamotTranslator,
    OpusmtTranslator,
    GoogleTranslator,
    MicrosoftTranslator,
    ArgosTranslator,
    NllbTranslator,
    BergamotPivotTranslator,
)

logger = get_logger(__file__)
logger.setLevel(logging.INFO)
logging.getLogger("argostranslate").disabled = True
logging.getLogger("argostranslate.utils").disabled = True

PIVOT_PAIRS = {("de", "fr"), ("fr", "de"), ("it", "de")}
ALL_METRICS = [Chrf, Chrfpp, Bleu, Comet22, MetricX24, Metricx24Qe, LlmRef]
ALL_DATASETS = [Flores200Plus, Wmt24pp, Bouquet]
ALL_TRANSLATORS = [
    BergamotTranslator,
    OpusmtTranslator,
    GoogleTranslator,
    MicrosoftTranslator,
    ArgosTranslator,
    NllbTranslator,
]
PROD_BUCKET = "moz-fx-translations-data--303e-prod-translations-data"


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
            "--storage",
            required=False,
            type=str,
            choices=["local", "gcs"],
            help="Storage type: local or gcs",
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
            choices=[d.name for d in ALL_DATASETS],
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
            self.storage = evals_config["storage"]
            self.languages = evals_config["languages"]
            self.datasets = evals_config["datasets"]
            self.translators = evals_config["translators"]
            self.metrics = evals_config["metrics"]
            self.models = evals_config["models"]
            self._validate()
        else:
            self.override = args.override
            self.storage = args.storage
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

        valid_datasets = [d.name for d in ALL_DATASETS]
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
    src_text: str
    trg_text: str
    ref_text: str

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

    def __init__(self, read_path: str, write_path: Path):
        # can be https://
        self.write_path = write_path
        self.read_path = read_path

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

    def _file_exists(self, path: Path) -> bool:
        return location_exists(f"{self.read_path}/{self._get_file_name(path)}")

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
            PROD_BUCKET if config.storage == "gcs" else config.artifacts_path,
            Path(config.artifacts_path),
        )
        self.run_timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        # Check if we're inside a Taskcluster task
        if os.environ.get("TASK_ID"):
            secrets = Secrets()
            secrets.prepare_keys()

        self.translators_cls = [
            t
            for t in ALL_TRANSLATORS
            if config.translators is None or t.name in config.translators
        ]
        self.metrics_cls = [
            m for m in ALL_METRICS if config.metrics is None or m.name in config.metrics
        ]  # type: list[type[RegularMetric]]
        self.datasets_cls = [
            d for d in ALL_DATASETS if config.datasets is None or d.name in config.datasets
        ]

    def run(self):
        if self.config.languages:
            lang_pairs = [lp.split("-") for lp in self.config.languages]
        else:
            bergamot_models = BergamotTranslator.list_all_models(PROD_BUCKET)
            lang_pairs = {(m.src, m.trg) for m in bergamot_models}.union(PIVOT_PAIRS)

        metrics_to_run = defaultdict(list)
        for src, trg in lang_pairs:
            logger.info(f"Running translators for {src} -> {trg}")
            for metric, metas in self.translate(src, trg).items():
                metrics_to_run[metric].extend(metas)

        logger.info(f"Running {len(metrics_to_run)} metrics for translations")
        self.run_metrics(metrics_to_run)

    def translate(self, src: str, trg: str) -> dict[type[RegularMetric], list[EvalsMeta]]:
        metrics_to_run = defaultdict(list[EvalsMeta])

        for dataset_cls in self.datasets_cls:
            if not dataset_cls.supports_lang(src, trg):
                logger.info(
                    f"Skipping dataset {dataset_cls.name}, it does not support {src} -> {trg}"
                )
                continue

            logger.info(f"Running for dataset {dataset_cls.name}")
            dataset = dataset_cls(src, trg)

            for translator_cls in self.translators_cls:
                model_names = None
                if translator_cls is BergamotTranslator:
                    if src != "en" and trg != "en" and translator_cls is BergamotTranslator:
                        translator_cls_new = BergamotPivotTranslator
                    else:
                        translator_cls_new = BergamotTranslator
                    translator = translator_cls_new(
                        src, trg, PROD_BUCKET, self.config.bergamot_cli_path
                    )

                    if self.config.models:
                        if len(self.config.models) == 1 and self.config.models[0] == "latest":
                            model_names = translator.list_latest_models()
                            logger.info(f"Using latest Bergamot model only {model_names}")
                        else:
                            model_names = [
                                m for m in translator.list_models() if m in self.config.models
                            ]
                            logger.info(f"Using specified Bergamot models {model_names}")
                else:
                    translator = translator_cls(src, trg)

                if not model_names:
                    model_names = translator.list_models()
                    logger.info(f"Using models {model_names}")

                for model_name in model_names:
                    meta = EvalsMeta(
                        src=src,
                        trg=trg,
                        dataset=dataset_cls.name,
                        translator=translator.name,
                        model_name=model_name,
                    )

                    translate = False
                    if not self.config.override:
                        for metric in self.metrics_cls:
                            if not self.storage.metric_exists(meta, metric.name):
                                metrics_to_run[metric].append(meta)
                                translate = True

                    if not translate:
                        logger.info(f"Skipping, metrics already exist for {meta.format_path()}")
                        continue

                    if not self.config.override and self.storage.translation_exists(meta):
                        logger.info(
                            f"Skipping translation, translations already exist for {meta.format_path()}"
                        )
                        continue

                    logger.info("Downloading dataset")
                    dataset.download()
                    # TODO: remove [:10] after testing
                    segments = dataset.get_texts()[:10]
                    source_texts = [s.source_text for s in segments]
                    ref_texts = [s.ref_text for s in segments]

                    logger.info(f"Running translator {translator.name}, model {model_name}")
                    translator.prepare(model_name)
                    logger.info(f"Translating {len(source_texts)} texts")
                    translations = translator.translate(source_texts)

                    to_save = [
                        Translation(s, t, r)
                        for s, t, r in zip(source_texts, translations, ref_texts)
                    ]
                    saved_path = self.storage.save_translations(meta, self.run_timestamp, to_save)
                    logger.info(f"Translations saved to {saved_path}")

                del translator
                self._clean()

        return metrics_to_run

    def run_metrics(self, to_run: dict[type[RegularMetric], list[EvalsMeta]]):
        for metric_cls, evals_metas in to_run.items():
            # Group by metric to loads some metrics on a GPU once
            logger.info(f"Running metric {metric_cls.name}")
            metric = metric_cls()

            for meta in evals_metas:
                if not metric.supports_lang(meta.src, meta.trg):
                    logger.info(
                        f"Skipping metric {metric_cls.name}, it does not support {meta.src} -> {meta.trg}"
                    )
                    continue

                translations = self.storage.load_translations(meta)
                source_texts, target_texts, ref_texts = zip(
                    *[(tr.src_text, tr.trg_text, tr.ref_text) for tr in translations]
                )

                logger.info(f"Scoring {len(ref_texts)} texts with {metric.name}")
                metric_results = metric.score(
                    meta.src, meta.trg, source_texts, target_texts, ref_texts
                )

                saved_path = self.storage.save_metric(meta, self.run_timestamp, metric_results)
                logger.info(f"Metric saved to {saved_path}")

            del metric
            self._clean()

    def _clean(self):
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    EvalsRunner(Config()).run()
