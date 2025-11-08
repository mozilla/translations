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
import json
import logging
import os
from dataclasses import dataclass
import datetime
from pathlib import Path

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
logger.setLevel(logging.DEBUG)

PIVOT_PAIRS = {("de", "fr"), ("fr", "de"), ("it", "de")}
ALL_METRICS = [Chrf, Chrfpp, Bleu, Comet22, MetricX24, Metricx24Qe]
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

    def __init__(self, read_path: str, write_path: Path):
        # can be https://
        self.write_path = write_path
        self.read_path = read_path

    def translation_exists(self, meta: EvalsMeta):
        return location_exists(
            f"{self.read_path}/{meta.format_path()}/{self.LATEST}/{self.TRANSLATIONS}"
        )

    def metrics_exists(self, meta: EvalsMeta):
        return location_exists(
            f"{self.read_path}/{meta.format_path()}/{self.LATEST}/{self.METRICS}"
        ) and location_exists(f"{self.read_path}/{meta.format_path()}/{self.LATEST}/{self.SCORES}")

    def save_translations(
        self, meta: EvalsMeta, timestamp: str, translations: list[Translation]
    ) -> Path:
        timestamp_path = meta.format_path() / timestamp
        json_obj = [tr.to_dict() for tr in translations]
        self._write(json_obj, timestamp_path / self.TRANSLATIONS)
        self._write(json_obj, meta.format_path() / self.LATEST / self.TRANSLATIONS)

        return timestamp_path

    def load_translations(self, meta: EvalsMeta, timestamp: str) -> list[Translation]:
        timestamp_path = self.write_path / meta.format_path() / timestamp / self.TRANSLATIONS
        with open(timestamp_path, "r", encoding="utf-8") as f:
            return [Translation.from_dict(tr) for tr in json.load(f)]

    def save_metrics(self, meta: EvalsMeta, timestamp: str, metrics: list[MetricResults]) -> Path:
        timestamp_path = meta.format_path() / timestamp

        metrics_json = {m.name: {"score": m.corpus_score, "details": m.details} for m in metrics}
        self._write(metrics_json, timestamp_path / self.METRICS)
        self._write(metrics_json, meta.format_path() / self.LATEST / self.METRICS)

        scores_json = {m.name: m.segment_scores for m in metrics}
        self._write(scores_json, timestamp_path / self.SCORES)
        self._write(scores_json, meta.format_path() / self.LATEST / self.SCORES)

        return timestamp_path

    def _write(self, data: object, path: Path):
        full_path = self.write_path / path
        os.makedirs(full_path.parent, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


class EvalsRunner:
    def __init__(self, config: Config):
        logger.info(f"Config: {config.print()}")
        self.config = config
        self.storage = Storage(
            PROD_BUCKET if config.storage == "gcs" else config.artifacts_path,
            Path(config.artifacts_path),
        )
        self.run_timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")

        self.translators_cls = [
            t
            for t in ALL_TRANSLATORS
            if config.translators is None or t.name in config.translators
        ]
        self.metrics_cls = [
            m for m in ALL_METRICS if config.metrics is None or m.name in config.metrics
        ]
        self.datasets_cls = [
            d for d in ALL_DATASETS if config.datasets is None or d.name in config.datasets
        ]

    def run(self):
        if self.config.languages:
            lang_pairs = [lp.split("-") for lp in self.config.languages]
        else:
            bergamot_models = BergamotTranslator.list_all_models(PROD_BUCKET)
            lang_pairs = {(m.src, m.trg) for m in bergamot_models}.union(PIVOT_PAIRS)

        metrics_to_run = []
        for src, trg in lang_pairs:
            logger.info(f"Running translators for {src} -> {trg}")
            metrics_to_run.extend(self.translate(src, trg))

        logger.info(f"Running metrics for {len(metrics_to_run)} translations")
        self.run_metrics(metrics_to_run)

    def translate(self, src: str, trg: str) -> list[EvalsMeta]:
        metrics_to_run = []

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

                    if not self.config.override and self.storage.translation_exists(meta):
                        logger.info(
                            f"Skipping, translations already exist for {meta.format_path()}"
                        )
                        continue

                    logger.info("Downloading dataset")
                    dataset.download()
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

                    if not self.config.override and self.storage.metrics_exists(meta):
                        logger.info(f"Skipping, metrics already exist for {meta.format_path()}")
                        continue

                    metrics_to_run.append(meta)

        return metrics_to_run

    def run_metrics(self, to_run: list[EvalsMeta]):
        all_results = []
        for metric_cls in self.metrics_cls:
            # Run by metrics to loads some metrics on a GPU once
            logger.info(f"Running metric {metric_cls.name}")
            metric = metric_cls()  # type: RegularMetric

            for meta in to_run:
                if not metric.supports_lang(meta.src, meta.trg):
                    logger.info(
                        f"Skipping metric {metric_cls.name}, it does not support {meta.src} -> {meta.trg}"
                    )
                    continue

                translations = self.storage.load_translations(meta, self.run_timestamp)
                source_texts, target_texts, ref_texts = zip(
                    *[(tr.src_text, tr.trg_text, tr.ref_text) for tr in translations]
                )

                logger.info(f"Scoring {len(ref_texts)} texts with {metric.name}")
                metric_results = metric.score(
                    meta.src, meta.trg, source_texts, target_texts, ref_texts
                )
                all_results.append(metric_results)

                saved_path = self.storage.save_metrics(meta, self.run_timestamp, all_results)
                logger.info(f"Metrics saved to {saved_path}")


if __name__ == "__main__":
    EvalsRunner(Config()).run()
