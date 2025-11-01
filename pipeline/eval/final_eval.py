from pipeline.common.downloads import location_exists
from pipeline.common.logging import get_logger
from pipeline.eval.datasets import Flores200Plus, Wmt24pp, Bouqet
from pipeline.eval.metrics import Chrfpp, Bleu
from pipeline.eval.translators import (
    BergamotTranslator,
    OpusmtTranslator,
    GoogleTranslator,
    MicrosoftTranslator,
    ArgosTranslator,
    NllbTranslator,
)

logger = get_logger(__file__)


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
    lang_pairs = {(m.src, m.trg) for m in bergamot_models}
    translators_cls = [
        BergamotTranslator,
        OpusmtTranslator,
        GoogleTranslator,
        MicrosoftTranslator,
        ArgosTranslator,
        NllbTranslator,
    ]
    metrics = [Chrfpp(), Bleu()]
    datasets = [Flores200Plus(), Wmt24pp(), Bouqet()]

    for src, trg in lang_pairs:
        for dataset in datasets:
            for translator_cls in translators_cls:
                translator = translator_cls(src, trg)

                for model_name in translator.list_models():
                    if storage.exists(src, trg, dataset.name, translator.name, model_name):
                        continue

                    dataset.download(src, trg)
                    translator.prepare(model_name)

                    segments = dataset.get_texts()
                    source_texts = [s.source_text for s in segments]
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
