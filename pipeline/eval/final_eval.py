import json
import os
from dataclasses import dataclass
import datetime
from pathlib import Path

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
LOCAL_STORAGE_PATH = "data/final_evals"
BERGAMOT_CLI_PATH = "inference/build/src/app/translator-cli"


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

    def __init__(self, base_path: Path):
        # can be https://
        self.base_path = base_path

    def translation_exists(self, meta: EvalsMeta):
        return location_exists(
            f"{self.base_path}/{meta.format_path()}/{self.LATEST}/{self.TRANSLATIONS}"
        )

    def metrics_exists(self, meta: EvalsMeta):
        return location_exists(
            f"{self.base_path}/{meta.format_path()}/{self.LATEST}/{self.METRICS}"
        ) and location_exists(f"{self.base_path}/{meta.format_path()}/{self.LATEST}/{self.SCORES}")

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
        ...


class GcsStorage(Storage):
    def _write(self, data: object, path: str):
        raise NotImplementedError()


class LocalStorage(Storage):
    def _write(self, data: object, path: Path):
        full_path = self.base_path / path
        os.makedirs(full_path.parent, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def run(
    src: str = None,
    trg: str = None,
    datasets: list[str] = None,
    translators: list[str] = None,
    metrics: list[str] = None,
):
    storage = LocalStorage("data/final_evals")
    run_timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")

    translators_cls = [t for t in ALL_TRANSLATORS if translators is None or t.name in translators]
    metrics_cls = [m for m in ALL_METRICS if metrics is None or m.name in metrics]
    datasets_cls = [d for d in ALL_DATASETS if datasets is None or d.name in datasets]

    if src and trg:
        lang_pairs = [(src, trg)]
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
            run_timestamp,
            storage,
            override=False,
        )


def run_lang_pair(
    src, trg, datasets_cls, translators_cls, metrics_cls, run_timestamp, storage, override: bool
):
    for dataset_cls in datasets_cls:
        if not dataset_cls.supports_lang(src, trg):
            logger.info(f"Skipping dataset {dataset_cls.name}, it does not support {src} -> {trg}")
            continue

        logger.info(f"Running for dataset {dataset_cls.name}")
        dataset = dataset_cls(src, trg)

        for translator_cls in translators_cls:
            if translator_cls is BergamotTranslator:
                if src != "en" and trg != "en" and translator_cls is BergamotTranslator:
                    translator_cls_new = BergamotPivotTranslator
                else:
                    translator_cls_new = BergamotTranslator
                translator = translator_cls_new(src, trg, PROD_BUCKET, BERGAMOT_CLI_PATH)
            else:
                translator = translator_cls(src, trg)

            for model_name in translator.list_models():
                meta = EvalsMeta(
                    src=src,
                    trg=trg,
                    dataset=dataset_cls.name,
                    translator=translator.name,
                    model_name=model_name,
                )

                if not override and storage.translation_exists(meta):
                    logger.info("Skipping, translations already exist")
                    continue

                logger.info("Downloading dataset")
                dataset.download()
                segments = dataset.get_texts()[:10]
                source_texts = [s.source_text for s in segments]

                logger.info(f"Running translator {translator.name}, model {model_name}")
                translator.prepare(model_name)
                logger.info(f"Translating {len(source_texts)} texts")
                translations = translator.translate(source_texts)
                saved_path = storage.save_translations(meta, run_timestamp, translations)
                logger.info(f"Translations saved to {saved_path}")

                if not override and storage.metrics_exists(meta):
                    logger.info("Skipping, metrics already exist")
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
    run(
        "ru",
        "de",
        metrics=["chrf", "chrfpp", "bleu"],
        datasets=["flores200-plus", "bouquet", "wmt24pp"],
        translators=["bergamot"],
    )
