"""
Run final evaluation for exported models and compare to other translators


"""
import argparse
import json
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
BERGAMOT_CLI_PATH = "inference/build/src/app/translator-cli"


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
        if args.config:
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

    def save_translations(self, meta: EvalsMeta, timestamp: str, translations: list[str]) -> Path:
        timestamp_path = meta.format_path() / timestamp
        self._write(translations, timestamp_path / self.TRANSLATIONS)
        self._write(translations, meta.format_path() / self.LATEST / self.TRANSLATIONS)

        return timestamp_path

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


def run(
    config: Config,
):
    logger.info(f"Config: {config.print()}")

    storage = Storage(
        PROD_BUCKET if config.storage == "gcs" else config.artifacts_path,
        Path(config.artifacts_path),
    )
    run_timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    translators_cls = [
        t for t in ALL_TRANSLATORS if config.translators is None or t.name in config.translators
    ]
    metrics_cls = [m for m in ALL_METRICS if config.metrics is None or m.name in config.metrics]
    datasets_cls = [
        d for d in ALL_DATASETS if config.datasets is None or d.name in config.datasets
    ]

    if config.languages:
        lang_pairs = [lp.split("-") for lp in config.languages]
    else:
        bergamot_models = BergamotTranslator.list_all_models(PROD_BUCKET)
        lang_pairs = {(m.src, m.trg) for m in bergamot_models}.union(PIVOT_PAIRS)

    for src, trg in lang_pairs:
        logger.info(f"Running for {src} -> {trg}")
        run_lang_pair(
            src,
            trg,
            datasets_cls,
            translators_cls,
            metrics_cls,
            config.models,
            run_timestamp,
            storage,
            config.override,
        )


def run_lang_pair(
    src,
    trg,
    datasets_cls,
    translators_cls,
    metrics_cls,
    models,
    run_timestamp,
    storage,
    override: bool,
):
    for dataset_cls in datasets_cls:
        if not dataset_cls.supports_lang(src, trg):
            logger.info(f"Skipping dataset {dataset_cls.name}, it does not support {src} -> {trg}")
            continue

        logger.info(f"Running for dataset {dataset_cls.name}")
        dataset = dataset_cls(src, trg)

        for translator_cls in translators_cls:
            model_names = None
            if translator_cls is BergamotTranslator:
                if src != "en" and trg != "en" and translator_cls is BergamotTranslator:
                    translator_cls_new = BergamotPivotTranslator
                else:
                    translator_cls_new = BergamotTranslator
                translator = translator_cls_new(src, trg, PROD_BUCKET, BERGAMOT_CLI_PATH)

                if models:
                    if len(models) == 1 and models[0] == "latest":
                        model_names = translator.list_latest_models()
                        logger.info(f"Using latest Bergamot model only {model_names}")
                    else:
                        model_names = [m for m in translator.list_models() if m in models]
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

                if not override and storage.translation_exists(meta):
                    logger.info(f"Skipping, translations already exist for {meta.format_path()}")
                    continue

                logger.info("Downloading dataset")
                dataset.download()
                segments = dataset.get_texts()
                source_texts = [s.source_text for s in segments]

                logger.info(f"Running translator {translator.name}, model {model_name}")
                translator.prepare(model_name)
                logger.info(f"Translating {len(source_texts)} texts")
                translations = translator.translate(source_texts)
                saved_path = storage.save_translations(meta, run_timestamp, translations)
                logger.info(f"Translations saved to {saved_path}")

                if not override and storage.metrics_exists(meta):
                    logger.info(f"Skipping, metrics already exist for {meta.format_path()}")
                    continue

                all_results = []
                for metric_cls in metrics_cls:
                    if not metric_cls.supports_lang(src, trg):
                        logger.info(
                            f"Skipping metric {metric_cls.name}, it does not support {src} -> {trg}"
                        )
                        continue
                    # loads some metrics on GPU
                    logger.info(f"Running metric {metric_cls.name}")
                    metric = metric_cls()
                    ref_texts = [s.ref_text for s in segments]
                    logger.info(f"Scoring {len(ref_texts)} texts")
                    metric_results = metric.score(src, trg, source_texts, translations, ref_texts)
                    all_results.append(metric_results)

                saved_path = storage.save_metrics(meta, run_timestamp, all_results)
                logger.info(f"Metrics saved to {saved_path}")


if __name__ == "__main__":
    run(Config())
