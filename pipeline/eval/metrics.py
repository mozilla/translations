from dataclasses import dataclass

import sacrebleu.metrics.base
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF


@dataclass
class MetricResults:
    name: str
    segment_scores: list[float]
    corpus_score: float
    details: str


class Metric:
    def supports_lang(self, src_lang: str, trg_lang: str) -> bool:
        pass


class RegularMetric(Metric):
    def score(
        self,
        src_lang: str,
        trg_lang: str,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str],
    ) -> MetricResults:
        pass


class ReferencelessMetric(Metric):
    def score(
        self, src_lang: str, trg_lang: str, source_texts: list[str], translated_texts: list[str]
    ) -> MetricResults:
        pass


class SacrebleuMetric(Metric):
    def __init__(self):
        self.metric: sacrebleu.metrics.base.Metric = None

    def supports_lang(self, src_lang: str, trg_lang: str) -> bool:
        return True

    def score(
        self,
        src_lang: str,
        trg_lang: str,
        source_texts: list[str],
        translated_texts: list[str],
        reference_texts: list[str],
    ) -> MetricResults:
        corpus_score = self.metric.corpus_score(translated_texts, [reference_texts])
        segment_scores = [
            self.metric.sentence_score(tr, [ref])
            for tr, ref in zip(translated_texts, reference_texts)
        ]

        return MetricResults(
            name=self.name,
            corpus_score=corpus_score.score,
            segment_scores=segment_scores,
            details=self.metric.get_signature().format(),
        )


class Chrf(SacrebleuMetric):
    name = "chrf"

    def __init__(self):
        super().__init__()
        self.metric = CHRF()


class Chrfpp(SacrebleuMetric):
    name = "chrfpp"

    def __init__(self):
        super().__init__()
        self.metric = CHRF(word_order=2)


class Bleu(SacrebleuMetric):
    name = "bleu"

    def __init__(self):
        # todo: double check what'up with the tokenizer
        super().__init__()
        self.metric = BLEU()

    def supports_lang(self, src_lang: str, trg_lang: str) -> bool:
        return True
