from pipeline.common.downloads import location_exists
from pipeline.common.logging import get_logger
from pipeline.eval.eval_datasets import Flores200Plus, Wmt24pp, Bouqet
from pipeline.eval.metrics import Chrfpp, Bleu, Chrf, Comet22, MetricX24, Metricx24Qe
from pipeline.eval.translators import (
    BergamotTranslator,
    OpusmtTranslator,
    GoogleTranslator,
    MicrosoftTranslator,
    ArgosTranslator,
    NllbTranslator,
)

logger = get_logger(__file__)

PIVOT_PAIRS = {("de", "fr"), ("fr", "de"), ("it", "de")}


class Storage:
    def __init__(self, bucket):
        self.bucket = bucket

    def exists(self, src: str, trg: str, dataset: str, translator: str, model_name: str):
        return location_exists(f"{self.bucket}/{src}-{trg}/{dataset}/{translator}/{model_name}")

    def save_translations(self, src, trg, translations, service, model_name):
        pass

    def save_metrics(self, src, trg, metric_results, service, model_name):
        pass


def run():
    bucket = ""
    storage = Storage(bucket)

    bergamot_models = BergamotTranslator.list_all_models(bucket)
    lang_pairs = {(m.src, m.trg) for m in bergamot_models}.union(PIVOT_PAIRS)
    translators_cls = [
        BergamotTranslator,
        OpusmtTranslator,
        GoogleTranslator,
        MicrosoftTranslator,
        ArgosTranslator,
        NllbTranslator,
    ]
    # todo: delay creation as some metrics load GPU
    metrics = [Chrf(), Chrfpp(), Bleu(), Comet22(), MetricX24(), Metricx24Qe()]
    datasets_cls = [Flores200Plus, Wmt24pp, Bouqet]

    for src, trg in lang_pairs:
        for dataset_cls in datasets_cls:
            if not dataset_cls.supports_lang(src, trg):
                continue
            dataset = dataset_cls(src, trg)

            for translator_cls in translators_cls:
                translator = translator_cls(src, trg)

                for model_name in translator.list_models():
                    if storage.exists(src, trg, dataset.name, translator.name, model_name):
                        continue

                    # todo: should download once
                    dataset.download()
                    segments = dataset.get_texts()
                    source_texts = [s.source_text for s in segments]

                    translator.prepare(model_name)

                    translations = translator.translate(source_texts)
                    storage.save_translations(src, trg, translations, translator.name, model_name)

                    for metric in metrics:
                        if not metric.supports_lang(src, trg):
                            continue
                        ref_texts = [s.ref_text for s in segments]
                        metric_results = metric.score(
                            src, trg, source_texts, translations, ref_texts
                        )
                        storage.save_metrics(src, trg, metric_results, translator.name, model_name)


if __name__ == "__main__":
    run()
