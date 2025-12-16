"""
Models for Translations DB entities
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

TASK_GROUP_ID_LENGTH = 22


@dataclass
class Evaluation:
    chrf: Optional[float] = None
    bleu: Optional[float] = None
    comet: Optional[float] = None


@dataclass
class Corpus:
    source_url: str
    source_bytes: int
    target_url: str
    target_bytes: int
    id: Optional[int] = None
    run_id: Optional[int] = None
    type: Optional[str] = None
    aligned: Optional[bool] = None


@dataclass
class WordAlignedCorpus:
    source_url: str
    target_url: str
    alignments_url: str
    source_bytes: int
    target_bytes: int
    alignments_bytes: int
    id: Optional[int] = None
    run_id: Optional[int] = None
    type: Optional[str] = None
    aligned: Optional[bool] = None


@dataclass
class Model:
    date: Optional[datetime] = None
    config: Optional[dict] = None
    task_group_id: Optional[str] = None
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    flores: Optional[Evaluation] = None
    artifact_folder: Optional[str] = None
    artifact_urls: list[str] = None
    id: Optional[int] = None
    run_id: Optional[int] = None
    kind: Optional[str] = None

    def __post_init__(self):
        if self.artifact_urls is None:
            self.artifact_urls = []


@dataclass
class TaskGroup:
    task_group_id: str
    run_id: Optional[int] = None
    experiment_config: Optional[dict] = None


@dataclass
class TrainingRun:
    name: str
    source_lang: str
    target_lang: str
    task_group_ids: list[str]
    date_created: Optional[datetime] = None
    experiment_config: Optional[dict] = None
    comet_flores_comparison: dict[str, float] = None
    bleu_flores_comparison: dict[str, float] = None

    parallel_corpus_aligned: Optional[WordAlignedCorpus] = None
    backtranslations_corpus_aligned: Optional[WordAlignedCorpus] = None
    distillation_corpus_aligned: Optional[WordAlignedCorpus] = None

    parallel_corpus: Optional[Corpus] = None
    backtranslations_corpus: Optional[Corpus] = None
    distillation_corpus: Optional[Corpus] = None

    backwards: Optional[Model] = None
    teacher_1: Optional[Model] = None
    teacher_2: Optional[Model] = None
    teacher_ensemble_flores: Optional[Evaluation] = None
    student: Optional[Model] = None
    student_finetuned: Optional[Model] = None
    student_quantized: Optional[Model] = None
    student_exported: Optional[Model] = None

    @property
    def langpair(self) -> str:
        return f"{self.source_lang}-{self.target_lang}"

    @property
    def date_started(self) -> Optional[datetime]:
        return self.date_created

    @date_started.setter
    def date_started(self, value: Optional[datetime]):
        self.date_created = value

    def __post_init__(self):
        if self.comet_flores_comparison is None:
            self.comet_flores_comparison = {}
        if self.bleu_flores_comparison is None:
            self.bleu_flores_comparison = {}

    @classmethod
    def create(cls, name: str, task_group_ids: list[str], langpair: str):
        source_lang, target_lang = langpair.split("-")
        return cls(
            name=name,
            source_lang=source_lang,
            target_lang=target_lang,
            task_group_ids=task_group_ids,
        )


@dataclass
class Task:
    task_id: str
    task_group_id: str
    created_date: str
    state: Optional[str] = None
    task_name: Optional[str] = None
    resolved_date: Optional[str] = None


@dataclass
class RunComparison:
    metric: str
    provider: str
    score: float


@dataclass
class FinalEval:
    source_lang: str
    target_lang: str
    dataset: str
    translator: str
    model_name: str
    translations_url: Optional[str] = None
    id: Optional[int] = None


@dataclass
class FinalEvalMetric:
    eval_id: int
    metric_name: str
    corpus_score: float
    details_json: Optional[str] = None
    scores_url: Optional[str] = None
    id: Optional[int] = None


@dataclass
class FinalEvalLlmScore:
    metric_id: int
    criterion: str
    score: float
    summary: Optional[str] = None
    id: Optional[int] = None


@dataclass
class Export:
    model_id: int
    architecture: str
    byte_size: int
    hash: str
    model_config: Optional[dict] = None
    model_statistics: Optional[dict] = None
    release_status: Optional[str] = None
    id: Optional[int] = None
